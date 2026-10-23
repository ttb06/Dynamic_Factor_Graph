#!/usr/bin/env python3
"""Same-backbone Harpoon-style differentiable guidance proxy.

This is not a Harpoon reproduction. It differentiates a squared hinge loss on
predicted clean numeric values with respect to the current noisy numeric state.
Only controlled monotone numeric contracts are covered.
"""
from __future__ import annotations
import argparse, copy, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import cross_dataset_condition_adapter as xc
from conditioned_tabdiff_adapter import decode, seed_all
from ctdf_style_cross_dataset import encoded_bounds, exact_project, target_distortion
from tabdiff_probe import preprocess_dataset


def sample_guided(diffusion, n: int, bounds: dict[int,float], eta: float, start_t: float) -> tuple[torch.Tensor,int]:
    device=diffusion.device
    times=torch.linspace(0,1,diffusion.num_timesteps,dtype=torch.float32,device=device)[:,None]
    sigma_num=diffusion.num_schedule.total_noise(times); sigma_cat=diffusion.cat_schedule.total_noise(times)
    sigma_num_next=torch.zeros_like(sigma_num); sigma_num_next[1:]=sigma_num[:-1]
    sigma_cat_next=torch.zeros_like(sigma_cat); sigma_cat_next[1:]=sigma_cat[:-1]
    if diffusion.sampler_params["stochastic_sampler"] or diffusion.sampler_params["second_order_correction"]:
        raise ValueError("proxy expects deterministic first-order sampling")
    numeric=torch.randn((n,diffusion.num_numerical_features),device=device)*sigma_num[-1]
    categorical=diffusion._sample_masked_prior(n,len(diffusion.num_classes)) if len(diffusion.num_classes) else torch.zeros((n,0),device=device)
    gradient_steps=0
    for index in reversed(range(diffusion.num_timesteps)):
        active=float(times[index])<=start_t
        with torch.enable_grad() if active else torch.no_grad():
            current=numeric.detach().requires_grad_(active)
            categorical_oh=diffusion.to_one_hot(categorical).to(current.dtype) if len(diffusion.num_classes) else categorical
            denoised,raw_logits=diffusion._denoise_fn(current.float(),categorical_oh,times[index].squeeze().repeat(n),sigma=sigma_num[index].unsqueeze(0).repeat(n,1))
            direction=(current-denoised)/sigma_num[index]
            numeric_next=current+(sigma_num_next[index]-sigma_num[index])*direction
            if len(diffusion.num_classes):
                logits=diffusion._subs_parameterization(raw_logits,categorical)
                alpha_t=torch.exp(-sigma_cat[index]).unsqueeze(0).repeat(n,1)
                alpha_s=torch.exp(-sigma_cat_next[index]).unsqueeze(0).repeat(n,1)
                categorical_next,_=diffusion._mdlm_update(logits,categorical,alpha_t,alpha_s)
            else:
                categorical_next=categorical
            if active:
                loss=torch.zeros(n,device=device)
                for column,bound in bounds.items():
                    loss=loss+torch.relu(torch.as_tensor(bound,device=device)-denoised[:,column]).square()
                gradient=torch.autograd.grad(loss.sum(),current,retain_graph=False,create_graph=False)[0]
                numeric_next=numeric_next.detach()-eta*gradient.detach()
                gradient_steps+=1
        numeric=numeric_next.detach(); categorical=categorical_next.detach()
    if len(diffusion.num_classes) and not torch.all(categorical<diffusion.mask_index): raise RuntimeError("categorical mask remained")
    return torch.cat([numeric,categorical],dim=1).cpu(),gradient_steps


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",type=Path,required=True); parser.add_argument("--dataset",choices=["default","shoppers"],required=True)
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=1024); parser.add_argument("--steps",type=int,default=30); parser.add_argument("--etas",default="0.01:0.03:0.1:0.3")
    parser.add_argument("--start-t",type=float,default=.75); parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    started=time.perf_counter(); device=torch.device("cuda"); seed_all(args.seed); sys.path.insert(0,str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model,UniModMLP
    info=json.loads((args.repo/"data"/args.dataset/"info.json").read_text()); train_data,_=preprocess_dataset(args.repo,args.dataset); real=decode(train_data,train_data.X,info)
    xc.CATALOGS_CURRENT=[info["column_names"][index] for index in info["num_col_idx"]]
    with args.config.open("rb") as handle: config=pickle.load(handle)
    params=copy.deepcopy(config["unimodmlp_params"]); params["d_numerical"]=train_data.d_numerical; params["categories"]=(train_data.categories+1).tolist()
    model=Model(UniModMLP(**params),**config["diffusion_params"]["edm_params"]); state=torch.load(args.checkpoint,map_location="cpu",weights_only=False); model.load_state_dict(state["denoise_fn"])
    dp=copy.deepcopy(config["diffusion_params"]); dp.update(num_timesteps=args.steps,scheduler="power_mean_per_column",cat_scheduler="log_linear_per_column")
    dp["sampler_params"]=copy.deepcopy(dp["sampler_params"]); dp["sampler_params"].update(stochastic_sampler=False,second_order_correction=False)
    diffusion=UnifiedCtimeDiffusion(train_data.categories,train_data.d_numerical,model,None,**dp,device=device).to(device)
    diffusion.num_schedule.load_state_dict(state["num_schedule"]); diffusion.cat_schedule.load_state_dict(state["cat_schedule"]); diffusion.eval()
    contracts=[contract for contract in xc.make_catalog(args.dataset) if contract["split"]!="train"]
    rows=[]
    for ordinal,contract in enumerate(contracts):
        common_seed=args.seed+10000+ordinal; seed_all(common_seed)
        with torch.no_grad(): base=decode(train_data,diffusion.sample(args.samples),info)
        base_rate=float(xc.mask(base,contract).mean()); metric=xc.evaluate(base,real,contract,base_rate,base_rate)
        metric.update(dataset=args.dataset,method="unconditional",eta=0.0,seed=args.seed,gradient_steps=0,edit_rate=0.0,target_distortion=0.0,runtime_seconds=0.0); rows.append(metric)
        bounds=encoded_bounds(train_data,real,info,contract)
        for eta in [float(value) for value in args.etas.replace(",",":").split(":") if value]:
            seed_all(common_seed); tick=time.perf_counter(); encoded,gradient_steps=sample_guided(diffusion,args.samples,bounds,eta,args.start_t); elapsed=time.perf_counter()-tick
            guided=decode(train_data,encoded,info); metric=xc.evaluate(guided,real,contract,base_rate,base_rate)
            metric.update(dataset=args.dataset,method="harpoon_style_proxy",eta=eta,seed=args.seed,gradient_steps=gradient_steps,edit_rate=0.0,target_distortion=target_distortion(guided,base,real,contract),runtime_seconds=elapsed); rows.append(metric)
            certified,edit_rate=exact_project(guided,contract); metric=xc.evaluate(certified,real,contract,base_rate,base_rate)
            metric.update(dataset=args.dataset,method="harpoon_style_proxy_then_certifier",eta=eta,seed=args.seed,gradient_steps=gradient_steps,edit_rate=edit_rate,target_distortion=target_distortion(certified,base,real,contract),runtime_seconds=elapsed); rows.append(metric)
    metrics=pd.DataFrame(rows); metrics.to_csv(args.output/"contract_metrics.csv",index=False)
    summary=metrics.groupby(["method","eta","split"]).agg(satisfaction=("satisfaction_rate","mean"),cohort_ks=("cohort_numeric_ks","mean"),categorical_tv=("cohort_categorical_tv","mean"),edit_rate=("edit_rate","mean"),target_distortion=("target_distortion","mean"),gradient_steps=("gradient_steps","mean"),runtime_seconds=("runtime_seconds","mean")).reset_index(); summary.to_csv(args.output/"summary.csv",index=False)
    metadata={"status":"executed","label":"Harpoon-style differentiable guidance proxy; not Harpoon reproduction","scope":"monotone numeric constraints only","dataset":args.dataset,"seed":args.seed,"samples_per_contract":args.samples,"steps":args.steps,"start_t":args.start_t,"runtime_seconds":time.perf_counter()-started}
    (args.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2)); print(summary.to_string(index=False)); print(json.dumps(metadata,indent=2),flush=True)

if __name__=="__main__": main()
