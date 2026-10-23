#!/usr/bin/env python3
"""Compare always-on and delayed row-applicability gates with paired noise."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from conditioned_tabdiff_adapter import ConditionedUniModMLP, decode, seed_all
from implication_condition_adapter import ApplicabilityGatedConditionedModel, clean, evaluate
from implication_contract_catalog import catalog, vector
from tabdiff_probe import preprocess_adult


def build(args, device):
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    contracts = catalog(); condition_dim = len(vector(contracts[0]))
    info = json.loads((args.repo / "data/adult/info.json").read_text())
    train_data, _ = preprocess_adult(args.repo)
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    backbone = ConditionedUniModMLP(condition_dim=condition_dim, **params)
    base_model = Model(backbone, **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = base_model.load_state_dict(checkpoint["denoise_fn"], strict=False)
    missing = [key for key in incompatible.missing_keys if "condition_projection" not in key]
    if incompatible.unexpected_keys or missing:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    backbone.mlp.condition_projection.load_state_dict(adapter["condition_projection"])
    backbone.mlp.mlp.load_state_dict(adapter["latent_mlp"])
    categorical_names = [info["column_names"][index] for index in info["target_col_idx"] + info["cat_col_idx"]]
    encoder_categories = train_data.cat_transform.named_steps["ordinal"].categories_
    category_codes = {
        name: {str(value).strip(): int(index) for index, value in enumerate(values)}
        for name, values in zip(categorical_names, encoder_categories)
    }
    category_positions = {name: index for index, name in enumerate(categorical_names)}
    numeric_names = [info["column_names"][index] for index in info["num_col_idx"]]
    numeric_positions = {name: index for index, name in enumerate(numeric_names)}
    conditioned = ApplicabilityGatedConditionedModel(
        base_model, backbone, condition_dim, train_data.categories, category_codes, category_positions,
        numeric_positions=numeric_positions,
    ).to(device)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(
        num_classes=train_data.categories, num_numerical_features=train_data.d_numerical,
        denoise_fn=conditioned, y_only_model=None, **diffusion_params, device=device,
    ).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    diffusion.eval()
    return contracts, info, train_data, conditioned, diffusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--gate-starts", default="1:0.5:0.25:0.1:-1")
    parser.add_argument("--include-global-early", action="store_true",
                        help="Also test global condition early, row-specific gate late.")
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    device = torch.device("cuda")
    seed_all(args.seed)
    started = time.perf_counter()
    contracts, info, train_data, conditioned, diffusion = build(args, device)
    real = clean(decode(train_data, train_data.X, info))

    baselines = {}
    conditioned.gate_start_t = -1
    for ordinal, contract in enumerate(contracts):
        condition = torch.tensor(vector(contract), dtype=torch.float32, device=device).repeat(args.samples, 1)
        conditioned.set_condition(condition, args.guidance)
        seed_all(args.seed + 10000 + ordinal)
        with torch.no_grad():
            frame = clean(decode(train_data, diffusion.sample(args.samples), info))
        baselines[contract["contract_id"]] = evaluate(frame, real, contract)

    rows = []
    starts = [float(value) for value in args.gate_starts.replace(",", ":").split(":") if value]
    schedule_specs = [("off", value) for value in starts]
    if args.include_global_early:
        schedule_specs += [("global", value) for value in starts if 0 <= value < 1]
    for early_policy, gate_start in schedule_specs:
        conditioned.gate_start_t = gate_start
        conditioned.early_policy = early_policy
        for ordinal, contract in enumerate(contracts):
            condition = torch.tensor(vector(contract), dtype=torch.float32, device=device).repeat(args.samples, 1)
            conditioned.set_condition(condition, args.guidance)
            seed_all(args.seed + 10000 + ordinal)
            with torch.no_grad():
                frame = clean(decode(train_data, diffusion.sample(args.samples), info))
            metric = evaluate(frame, real, contract, baselines[contract["contract_id"]])
            if early_policy == "global":
                schedule = f"global_then_gate_t_le_{gate_start:g}"
            else:
                schedule = "never" if gate_start < 0 else ("always" if gate_start >= 1 else f"t_le_{gate_start:g}")
            metric.update(seed=args.seed, guidance_scale=args.guidance, gate_start_t=gate_start,
                          early_policy=early_policy, schedule=schedule)
            rows.append(metric)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "schedule_metrics.csv", index=False)
    summary = metrics.groupby(["schedule", "gate_start_t", "split"]).agg(
        contracts=("contract_id", "size"), conditional_violation=("conditional_violation_rate", "mean"),
        violation_reduction=("conditional_violation_reduction", "mean"), antecedent_error=("antecedent_rate_error", "mean"),
        exception_error=("exception_rate_error", "mean"), inactive_target_error=("inactive_target_mean_error", "mean"),
        numeric_ks=("global_numeric_ks", "mean"), categorical_tv=("global_categorical_tv", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "status": "executed", "seed": args.seed, "samples_per_contract": args.samples,
        "steps": args.steps, "guidance": args.guidance, "gate_starts": starts,
        "adapter": str(args.adapter), "runtime_seconds": time.perf_counter() - started,
        "include_global_early": args.include_global_early,
        "scope": "paired-noise inference schedule probe; no retraining",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
