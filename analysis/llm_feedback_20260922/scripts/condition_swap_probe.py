#!/usr/bin/env python3
"""Paired-noise causal test of whether adapter conditions encode request meaning."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditioned_tabdiff_adapter import ConditionedUniModMLP, ContractConditionedModel, decode, seed_all
from contract_condition_catalog import catalog
from tabdiff_probe import preprocess_adult


PAIRS = [
    ("age_ge_30", "age_ge_45", "age", 30.0, 45.0),
    ("hours_ge_30", "hours_ge_45", "hours.per.week", 30.0, 45.0),
    ("edu_ge_10", "edu_ge_13", "education.num", 10.0, 13.0),
]
TRAIN_PAIRS = [
    ("age_ge_25", "age_ge_50", "age", 25.0, 50.0),
    ("hours_ge_20", "hours_ge_50", "hours.per.week", 20.0, 50.0),
    ("edu_ge_9", "edu_ge_14", "education.num", 9.0, 14.0),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--pair-set", choices=["heldout", "train", "all"], default="heldout")
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    device = torch.device("cuda"); seed_all(args.seed); sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    representation, condition_dim = adapter["representation"], int(adapter["condition_dim"])
    vectors = {key: np.asarray(value, dtype=np.float32) for key, value in adapter["vectors"].items()}
    contracts = {item["contract_id"]: item for item in catalog()}
    train_vectors = [vectors[item["contract_id"]] for item in contracts.values() if item["split"] == "train"]
    generic = np.mean(train_vectors, axis=0)
    info = json.loads((args.repo / "data/adult/info.json").read_text())
    train_data, _ = preprocess_adult(args.repo)
    with args.config.open("rb") as handle: config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical; params["categories"] = (train_data.categories + 1).tolist()
    backbone = ConditionedUniModMLP(condition_dim=condition_dim, **params)
    base = Model(backbone, **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = base.load_state_dict(checkpoint["denoise_fn"], strict=False)
    if incompatible.unexpected_keys or [key for key in incompatible.missing_keys if "condition_projection" not in key]:
        raise RuntimeError(str(incompatible))
    backbone.mlp.condition_projection.load_state_dict(adapter["condition_projection"])
    backbone.mlp.mlp.load_state_dict(adapter["latent_mlp"])
    model = ContractConditionedModel(base, backbone, condition_dim).to(device)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(
        num_classes=train_data.categories, num_numerical_features=train_data.d_numerical,
        denoise_fn=model, y_only_model=None, **diffusion_params, device=device,
    ).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"]); diffusion.eval()

    variants = ["correct", "swapped", "null", "generic"]
    if representation == "hybrid": variants += ["ast_only", "text_only"]
    rows = []
    numeric_columns = [info["column_names"][index] for index in info["num_col_idx"]]
    categorical_columns = [info["column_names"][index] for index in info["cat_col_idx"] + info["target_col_idx"]]
    pairs = PAIRS if args.pair_set == "heldout" else TRAIN_PAIRS if args.pair_set == "train" else TRAIN_PAIRS + PAIRS
    for pair_index, (low_id, high_id, target, low_threshold, high_threshold) in enumerate(pairs):
        low_correct, high_correct = vectors[low_id].copy(), vectors[high_id].copy()
        for variant in variants:
            if variant == "correct": low_vector, high_vector = low_correct, high_correct
            elif variant == "swapped": low_vector, high_vector = high_correct, low_correct
            elif variant == "null": low_vector = high_vector = np.zeros(condition_dim, dtype=np.float32)
            elif variant == "generic": low_vector = high_vector = generic
            elif variant == "ast_only":
                low_vector, high_vector = low_correct.copy(), high_correct.copy()
                low_vector[6:] = 0; high_vector[6:] = 0
            elif variant == "text_only":
                low_vector, high_vector = low_correct.copy(), high_correct.copy()
                low_vector[:6] = 0; high_vector[:6] = 0
            generated = []
            for side, condition in (("low", low_vector), ("high", high_vector)):
                tensor = torch.tensor(condition, device=device).float().repeat(args.samples, 1)
                model.set_condition(tensor, args.guidance)
                seed_all(args.seed + 1000 + pair_index)
                with torch.no_grad(): encoded = diffusion.sample(args.samples)
                generated.append((side, encoded, decode(train_data, encoded, info)))
            low_encoded, low = generated[0][1], generated[0][2]
            high_encoded, high = generated[1][1], generated[1][2]
            low_target = pd.to_numeric(low[target], errors="coerce").to_numpy()
            high_target = pd.to_numeric(high[target], errors="coerce").to_numpy()
            other_differences = []
            for column in numeric_columns:
                if column == target: continue
                scale = max(float(pd.to_numeric(low[column]).std()), 1.0)
                other_differences.append(float(np.mean(np.abs(pd.to_numeric(high[column]) - pd.to_numeric(low[column]))) / scale))
            cat_change = np.mean([
                (high[column].astype(str).to_numpy() != low[column].astype(str).to_numpy()).mean()
                for column in categorical_columns
            ])
            row = {
                "representation": representation, "seed": args.seed, "guidance": args.guidance,
                "pair": f"{low_id}__{high_id}", "target": target, "variant": variant,
                "low_target_mean": float(low_target.mean()), "high_target_mean": float(high_target.mean()),
                "semantic_margin_high_minus_low": float(high_target.mean() - low_target.mean()),
                "paired_strict_monotonic_rate": float((high_target > low_target).mean()),
                "paired_non_decreasing_rate": float((high_target >= low_target).mean()),
                "low_request_satisfaction": float((low_target >= low_threshold).mean()),
                "high_request_satisfaction": float((high_target >= high_threshold).mean()),
                "mean_other_numeric_change_std_units": float(np.mean(other_differences)),
                "categorical_change_rate": float(cat_change),
                "encoded_row_l2": float(torch.linalg.vector_norm(high_encoded - low_encoded, dim=1).mean()),
            }
            rows.append(row); print(json.dumps(row), flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(args.output / "swap_metrics.csv", index=False)
    summary = frame.groupby(["representation", "variant"]).agg(
        pairs=("pair", "size"), semantic_margin=("semantic_margin_high_minus_low", "mean"),
        strict_monotonic=("paired_strict_monotonic_rate", "mean"), high_satisfaction=("high_request_satisfaction", "mean"),
        other_numeric_change=("mean_other_numeric_change_std_units", "mean"), categorical_change=("categorical_change_rate", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "status": "executed", "representation": representation, "seed": args.seed,
        "samples": args.samples, "steps": args.steps, "guidance": args.guidance,
        "pair_set": args.pair_set,
        "adapter": str(args.adapter),
        "interpretation": "Correct-vs-swapped/null/generic uses paired sampling noise; this tests causal condition semantics, not overall model quality.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
