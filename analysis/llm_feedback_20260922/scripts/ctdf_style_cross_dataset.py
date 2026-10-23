#!/usr/bin/env python3
"""Same-backbone CTDF-style numeric feasibility proxy.

This is not a CTDF reproduction: no official implementation was available,
and this proxy covers monotone numeric constraints only (not categorical KL
projection). It applies an encoded-space clamp after every reverse step or in
the late half, with paired initial noise against unconditional TabDiff.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import cross_dataset_condition_adapter as xc
from conditioned_tabdiff_adapter import decode, seed_all
from tabdiff_probe import preprocess_dataset


@torch.no_grad()
def sample_projected(diffusion, num_samples: int, lower_bounds: dict[int, float], start_t: float) -> torch.Tensor:
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num_cur = diffusion.num_schedule.total_noise(times)
    sigma_cat_cur = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num_cur); sigma_num_next[1:] = sigma_num_cur[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat_cur); sigma_cat_next[1:] = sigma_cat_cur[:-1]
    # All submitted proxy jobs use the deterministic first-order sampler.
    if diffusion.sampler_params["stochastic_sampler"] or diffusion.sampler_params["second_order_correction"]:
        raise ValueError("proxy expects deterministic first-order sampling")
    sigma_num_hat = sigma_num_cur; sigma_cat_hat = sigma_cat_cur; time_hat = times
    numeric = torch.randn((num_samples, diffusion.num_numerical_features), device=device) * sigma_num_cur[-1]
    categorical = torch.zeros((num_samples, 0), device=device).float()
    if len(diffusion.num_classes) > 0:
        categorical = diffusion._sample_masked_prior(num_samples, len(diffusion.num_classes))
    for index in reversed(range(diffusion.num_timesteps)):
        numeric, categorical, _ = diffusion.edm_update(
            numeric, categorical, index,
            times[index], times[index - 1] if index > 0 else None, time_hat[index],
            sigma_num_cur[index], sigma_num_next[index], sigma_num_hat[index],
            sigma_cat_cur[index], sigma_cat_next[index], sigma_cat_hat[index],
        )
        if float(times[index]) <= start_t:
            for column, value in lower_bounds.items():
                numeric[:, column].clamp_(min=value)
    if not torch.all(categorical < diffusion.mask_index):
        raise RuntimeError("categorical mask remained at final step")
    return torch.cat([numeric, categorical], dim=1).cpu()


def encoded_bounds(train_data, real: pd.DataFrame, info: dict, contract: dict) -> dict[int, float]:
    numeric_names = [info["column_names"][index] for index in info["num_col_idx"]]
    raw = real[numeric_names].median(numeric_only=True).to_numpy(dtype=float)[None, :]
    positions = {name:index for index,name in enumerate(numeric_names)}
    bounds = {}
    for column, threshold in contract["predicates"]:
        probe = raw.copy(); probe[0, positions[column]] = threshold
        transformed = train_data.num_transform.transform(probe)
        bounds[positions[column]] = float(transformed[0, positions[column]])
    return bounds


def target_distortion(sample: pd.DataFrame, base: pd.DataFrame, real: pd.DataFrame, contract: dict) -> float:
    values = []
    for column, _ in contract["predicates"]:
        scale = max(float(pd.to_numeric(real[column]).std()), 1e-6)
        values.append(float((pd.to_numeric(sample[column]) - pd.to_numeric(base[column])).abs().mean() / scale))
    return float(np.mean(values))


def exact_project(frame: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, float]:
    """Certify raw-space monotone constraints after inverse transformation."""
    projected = frame.copy()
    edited = np.zeros(len(projected), dtype=bool)
    for column, threshold in contract["predicates"]:
        bad = pd.to_numeric(projected[column]).lt(threshold)
        projected.loc[bad, column] = threshold
        edited |= bad.to_numpy()
    return projected, float(edited.mean())


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",type=Path,required=True); parser.add_argument("--dataset",choices=["default","shoppers"],required=True)
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=1024); parser.add_argument("--steps",type=int,default=30); parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    started=time.perf_counter(); device=torch.device("cuda"); seed_all(args.seed); sys.path.insert(0,str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP
    info=json.loads((args.repo/"data"/args.dataset/"info.json").read_text()); train_data,_=preprocess_dataset(args.repo,args.dataset); real=decode(train_data,train_data.X,info)
    xc.CATALOGS_CURRENT=[info["column_names"][index] for index in info["num_col_idx"]]
    with args.config.open("rb") as handle: config=pickle.load(handle)
    params=copy.deepcopy(config["unimodmlp_params"]); params["d_numerical"]=train_data.d_numerical; params["categories"]=(train_data.categories+1).tolist()
    model=Model(UniModMLP(**params),**config["diffusion_params"]["edm_params"]); state=torch.load(args.checkpoint,map_location="cpu",weights_only=False); model.load_state_dict(state["denoise_fn"])
    diffusion_params=copy.deepcopy(config["diffusion_params"]); diffusion_params.update(num_timesteps=args.steps,scheduler="power_mean_per_column",cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"]=copy.deepcopy(diffusion_params["sampler_params"]); diffusion_params["sampler_params"].update(stochastic_sampler=False,second_order_correction=False)
    diffusion=UnifiedCtimeDiffusion(train_data.categories,train_data.d_numerical,model,None,**diffusion_params,device=device).to(device)
    diffusion.num_schedule.load_state_dict(state["num_schedule"]); diffusion.cat_schedule.load_state_dict(state["cat_schedule"]); diffusion.eval()
    rows=[]
    for ordinal,contract in enumerate(xc.make_catalog(args.dataset)):
        common_seed=args.seed+10000+ordinal; seed_all(common_seed)
        with torch.no_grad(): base=decode(train_data,diffusion.sample(args.samples),info)
        base_rate=float(xc.mask(base,contract).mean())
        metric=xc.evaluate(base,real,contract,base_rate,base_rate); metric.update(dataset=args.dataset,method="unconditional",seed=args.seed,passes=1,projection_steps=0,target_distortion=0.0,edit_rate=0.0); rows.append(metric)
        bounds=encoded_bounds(train_data,real,info,contract)
        for method,start_t in (("ctdf_style_all_steps",1.0),("ctdf_style_late_half",0.5),("ctdf_style_last_step",0.0)):
            seed_all(common_seed)
            projected=decode(train_data,sample_projected(diffusion,args.samples,bounds,start_t),info)
            metric=xc.evaluate(projected,real,contract,base_rate,base_rate)
            steps=sum(float(value)<=start_t for value in torch.linspace(0,1,args.steps))
            metric.update(dataset=args.dataset,method=method,seed=args.seed,passes=1,projection_steps=steps,target_distortion=target_distortion(projected,base,real,contract),edit_rate=np.nan); rows.append(metric)
            if method == "ctdf_style_all_steps":
                certified, residual_edit_rate = exact_project(projected, contract)
                metric=xc.evaluate(certified,real,contract,base_rate,base_rate)
                metric.update(dataset=args.dataset,method="ctdf_style_all_steps_certified",seed=args.seed,passes=1,projection_steps=steps+1,target_distortion=target_distortion(certified,base,real,contract),edit_rate=residual_edit_rate); rows.append(metric)
        repaired, edit_rate = exact_project(base, contract)
        metric=xc.evaluate(repaired,real,contract,base_rate,base_rate); metric.update(dataset=args.dataset,method="raw_final_projection",seed=args.seed,passes=1,projection_steps=1,target_distortion=target_distortion(repaired,base,real,contract),edit_rate=edit_rate); rows.append(metric)
    metrics=pd.DataFrame(rows); metrics.to_csv(args.output/"contract_metrics.csv",index=False)
    summary=metrics.groupby(["method","split"]).agg(satisfaction=("satisfaction_rate","mean"),cohort_ks=("cohort_numeric_ks","mean"),categorical_tv=("cohort_categorical_tv","mean"),target_distortion=("target_distortion","mean"),edit_rate=("edit_rate","mean"),projection_steps=("projection_steps","mean"),passes=("passes","mean")).reset_index(); summary.to_csv(args.output/"summary.csv",index=False)
    metadata={"status":"executed","dataset":args.dataset,"seed":args.seed,"samples_per_contract":args.samples,"steps":args.steps,"label":"CTDF-style numeric projection proxy; not CTDF reproduction","scope":"monotone numeric lower bounds only; no categorical KL feasibility","runtime_seconds":time.perf_counter()-started}
    (args.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2)); print(summary.to_string(index=False)); print(json.dumps(metadata,indent=2),flush=True)


if __name__=="__main__": main()
