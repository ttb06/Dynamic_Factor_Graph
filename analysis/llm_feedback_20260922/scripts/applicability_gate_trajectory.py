#!/usr/bin/env python3
"""Audit when a per-row applicability gate becomes reliable during sampling."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditioned_tabdiff_adapter import ConditionedUniModMLP, decode, seed_all
from implication_condition_adapter import ApplicabilityGatedConditionedModel, clean
from implication_contract_catalog import antecedent_mask, catalog, exception_mask, vector
from tabdiff_probe import preprocess_adult


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def time_bin(t: float) -> str:
    if t > 0.75:
        return "(0.75,1.0]"
    if t > 0.50:
        return "(0.50,0.75]"
    if t > 0.25:
        return "(0.25,0.50]"
    if t > 0.10:
        return "(0.10,0.25]"
    return "[0,0.10]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scales", default="1,2,4")
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    device = torch.device("cuda")
    seed_all(args.seed)
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    contracts = catalog()
    condition_dim = len(vector(contracts[0]))
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
    conditioned = ApplicabilityGatedConditionedModel(
        base_model, backbone, condition_dim, train_data.categories, category_codes, category_positions,
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

    rows = []
    guidance_values = [float(value) for value in args.guidance_scales.replace(":", ",").split(",") if value]
    for guidance in guidance_values:
        for ordinal, contract in enumerate(contracts):
            condition = torch.tensor(vector(contract), dtype=torch.float32, device=device).repeat(args.samples, 1)
            conditioned.set_condition(condition, guidance)
            conditioned.start_trace()
            seed_all(args.seed + 10000 + ordinal)
            with torch.no_grad():
                encoded = diffusion.sample(args.samples)
            history = conditioned.stop_trace()
            frame = clean(decode(train_data, encoded, info))
            final_gate = (antecedent_mask(frame, contract) & ~exception_mask(frame, contract)).to_numpy(dtype=bool)
            final_codes = encoded[:, train_data.d_numerical:].long().cpu()
            relevant = []
            if vector(contract)[0] > 0.5:
                relevant.append(category_positions["education"])
            if vector(contract)[1] > 0.5:
                relevant.append(category_positions["workclass"])
            if vector(contract)[5] > 0.5:
                relevant.append(category_positions["occupation"])
            previous = None
            for call_index, trace in enumerate(history):
                gate = trace["gate"].numpy()
                true_positive = int(np.logical_and(gate, final_gate).sum())
                true_negative = int(np.logical_and(~gate, ~final_gate).sum())
                relevant_agreement = (
                    float(trace["codes"][:, relevant].eq(final_codes[:, relevant]).float().mean())
                    if relevant else math.nan
                )
                rows.append(
                    {
                        "seed": args.seed, "guidance_scale": guidance,
                        "contract_id": contract["contract_id"], "split": contract["split"],
                        "call_index": call_index, "t": trace["t"], "time_bin": time_bin(trace["t"]),
                        "gate_rate": float(gate.mean()), "final_applicable_rate": float(final_gate.mean()),
                        "gate_accuracy_to_final": float((gate == final_gate).mean()),
                        "gate_precision_to_final": safe_ratio(true_positive, int(gate.sum())),
                        "gate_recall_to_final": safe_ratio(true_positive, int(final_gate.sum())),
                        "gate_specificity_to_final": safe_ratio(true_negative, int((~final_gate).sum())),
                        "flip_rate_from_previous": math.nan if previous is None else float((gate != previous).mean()),
                        "all_category_agreement_to_final": float(trace["codes"].eq(final_codes).float().mean()),
                        "relevant_category_agreement_to_final": relevant_agreement,
                    }
                )
                previous = gate
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "gate_trajectory.csv", index=False)
    summary = metrics.groupby(["guidance_scale", "split", "time_bin"], sort=False).agg(
        contracts=("contract_id", "nunique"), gate_accuracy=("gate_accuracy_to_final", "mean"),
        gate_precision=("gate_precision_to_final", "mean"), gate_recall=("gate_recall_to_final", "mean"),
        flip_rate=("flip_rate_from_previous", "mean"),
        relevant_category_agreement=("relevant_category_agreement_to_final", "mean"),
        all_category_agreement=("all_category_agreement_to_final", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary_by_time_bin.csv", index=False)
    metadata = {
        "status": "executed", "seed": args.seed, "samples_per_contract": args.samples,
        "steps": args.steps, "guidance_scales": guidance_values, "adapter": str(args.adapter),
        "interpretation": "Gate labels are compared with the final generated row, not real data or an LLM judgement.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
