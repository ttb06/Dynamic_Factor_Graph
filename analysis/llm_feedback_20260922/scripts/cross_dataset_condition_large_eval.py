#!/usr/bin/env python3
"""Large-sample evaluation of an already trained structured-condition adapter."""
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
from conditioned_tabdiff_adapter import ConditionedUniModMLP, ContractConditionedModel, decode, seed_all
from tabdiff_probe import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", choices=["default", "shoppers"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scales", default="2:4")
    parser.add_argument("--include-final-projection", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    started = time.perf_counter()
    device = torch.device("cuda")
    seed_all(args.seed)
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, _ = preprocess_dataset(args.repo, args.dataset)
    real = decode(train_data, train_data.X, info)
    xc.CATALOGS_CURRENT = [info["column_names"][index] for index in info["num_col_idx"]]
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    saved = torch.load(args.adapter, map_location="cpu", weights_only=False)
    condition_dim = int(saved["condition_dim"])
    vectors = saved["vectors"]
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    backbone = ConditionedUniModMLP(condition_dim=condition_dim, **params)
    base = Model(backbone, **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = base.load_state_dict(checkpoint["denoise_fn"], strict=False)
    missing = [key for key in incompatible.missing_keys if "condition_projection" not in key]
    if incompatible.unexpected_keys or missing:
        raise RuntimeError(str(incompatible))
    backbone.mlp.condition_projection.load_state_dict(saved["condition_projection"])
    backbone.mlp.mlp.load_state_dict(saved["latent_mlp"])
    model = ContractConditionedModel(base, backbone, condition_dim).to(device)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, model, None,
        **diffusion_params, device=device,
    ).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    diffusion.eval()

    contracts = [contract for contract in xc.make_catalog(args.dataset) if contract["split"] != "train"]
    model.set_condition(torch.zeros(args.samples, condition_dim, device=device), 1)
    seed_all(args.seed + 9000)
    with torch.no_grad():
        null = decode(train_data, diffusion.sample(args.samples), info)
    null_rates = {contract["contract_id"]: float(xc.mask(null, contract).mean()) for contract in contracts}
    rows = []
    for guidance in [float(value) for value in args.guidance_scales.replace(",", ":").split(":") if value]:
        for ordinal, contract in enumerate(contracts):
            condition = torch.tensor(vectors[contract["contract_id"]], dtype=torch.float32, device=device).repeat(args.samples, 1)
            model.set_condition(condition, guidance)
            seed_all(args.seed + 10000 + ordinal)
            with torch.no_grad():
                sample = decode(train_data, diffusion.sample(args.samples), info)
            metric = xc.evaluate(sample, real, contract, null_rates[contract["contract_id"]], null_rates[contract["contract_id"]])
            metric.update(dataset=args.dataset, representation="structured", method="structured_condition",
                          guidance_scale=guidance, seed=args.seed, edit_rate=0.0, target_distortion=0.0)
            rows.append(metric)
            if args.include_final_projection:
                projected = sample.copy()
                edited = np.zeros(len(projected), dtype=bool)
                distortions = []
                for column, threshold in contract["predicates"]:
                    old = pd.to_numeric(projected[column]).copy()
                    bad = old.lt(threshold)
                    projected.loc[bad, column] = threshold
                    edited |= bad.to_numpy()
                    scale = max(float(pd.to_numeric(real[column]).std()), 1e-6)
                    distortions.append(float((pd.to_numeric(projected[column]) - old).abs().mean() / scale))
                projected_metric = xc.evaluate(
                    projected, real, contract, null_rates[contract["contract_id"]], null_rates[contract["contract_id"]]
                )
                projected_metric.update(
                    dataset=args.dataset, representation="structured", method="structured_then_final_projection",
                    guidance_scale=guidance, seed=args.seed, edit_rate=float(edited.mean()),
                    target_distortion=float(np.mean(distortions)),
                )
                rows.append(projected_metric)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "contract_metrics.csv", index=False)
    summary = metrics.groupby(["method", "guidance_scale", "split"]).agg(
        contracts=("contract_id", "size"), satisfaction=("satisfaction_rate", "mean"),
        gain_null=("gain_null", "mean"), cohort_ks=("cohort_numeric_ks", "mean"),
        categorical_tv=("cohort_categorical_tv", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "status":"executed", "dataset":args.dataset, "seed":args.seed,
        "samples_per_contract":args.samples, "steps":args.steps, "adapter":str(args.adapter),
        "scope":"evaluation-only; no retraining", "runtime_seconds":time.perf_counter() - started,
        "include_final_projection":args.include_final_projection,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
