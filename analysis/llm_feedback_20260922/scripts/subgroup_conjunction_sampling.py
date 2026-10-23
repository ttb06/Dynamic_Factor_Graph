#!/usr/bin/env python3
"""One-seed screen for subgroup-conditioned numeric conjunctions."""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

from conditioned_tabdiff_adapter import ConditionedUniModMLP, ContractConditionedModel, decode, seed_all, total_variation
from mixed_type_condition_adapter import structured_vector
from subgroup_conditioned_sampling import (
    Policy, masks, original_numeric_position, sample_conditioned, sample_unconditional_trace,
)
from tabdiff_probe import preprocess_dataset


@dataclass(frozen=True)
class Conjunction:
    name: str
    context: Policy
    constraints: tuple[tuple[str, float], ...]


def conjunctions(dataset: str) -> list[Conjunction]:
    if dataset == "default":
        return [
            Conjunction("pay2_two_payments", Policy("ctx", "PAY_2", "0", "BILL_AMT3", "PAY_AMT3", 500, "conjunction"),
                        (("PAY_AMT3", 500.0), ("PAY_AMT4", 500.0))),
            Conjunction("pay5_two_payments", Policy("ctx", "PAY_5", "-1", "BILL_AMT5", "PAY_AMT4", 1000, "conjunction"),
                        (("PAY_AMT4", 1000.0), ("PAY_AMT5", 1000.0))),
            Conjunction("pay6_bill_payment", Policy("ctx", "PAY_6", "-1", "BILL_AMT6", "PAY_AMT5", 2000, "conjunction"),
                        (("PAY_AMT5", 2000.0), ("BILL_AMT5", 2000.0))),
        ]
    return [
        Conjunction("returning_weekend_value_product",
                    Policy("ctx", "VisitorType", "Returning_Visitor", "Weekend", "PageValues", 5, "conjunction", "True"),
                    (("PageValues", 5.0), ("ProductRelated_Duration", 1500.0))),
        Conjunction("november_returning_product_value",
                    Policy("ctx", "Month", "Nov", "VisitorType", "ProductRelated_Duration", 1500, "conjunction", "Returning_Visitor"),
                    (("ProductRelated_Duration", 1500.0), ("PageValues", 5.0))),
        Conjunction("may_weekend_admin_product",
                    Policy("ctx", "Month", "May", "Weekend", "Administrative_Duration", 150, "conjunction", "True"),
                    (("Administrative_Duration", 150.0), ("ProductRelated_Duration", 600.0))),
    ]


def validity(frame: pd.DataFrame, item: Conjunction) -> np.ndarray:
    result = np.ones(len(frame), dtype=bool)
    for column, threshold in item.constraints:
        result &= pd.to_numeric(frame[column], errors="coerce").ge(threshold).fillna(False).to_numpy()
    return result


def certify(frame: pd.DataFrame, item: Conjunction, gate: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    for column, threshold in item.constraints:
        bad = gate & pd.to_numeric(result[column], errors="coerce").lt(threshold).fillna(True).to_numpy()
        result.loc[bad, column] = threshold
    return result


def numeric_scope(dataset: str, info: dict, item: Conjunction) -> list[str]:
    names = [info["column_names"][index] for index in info["num_col_idx"]]
    if dataset == "default":
        names = [name for name in names if (name.startswith("BILL_AMT") or name.startswith("PAY_AMT"))
                 and name != item.context.subgroup]
    return names


@torch.no_grad()
def bestof(diffusion, conditioned, train_data, info: dict, base_encoded: torch.Tensor,
           base_frame: pd.DataFrame, bank: pd.DataFrame, item: Conjunction,
           violated: np.ndarray, output_names: list[str], seed: int, branches: int = 8):
    result = base_frame.copy(); indices = np.flatnonzero(violated)
    if not len(indices):
        return result, 1.0
    positions = [original_numeric_position(info, name) for name in output_names]
    subset = base_encoded[indices]
    conditioned.set_numeric_output_mask(None)
    conditioned.set_condition(torch.zeros((len(indices), conditioned.condition_dim), device=diffusion.device), 1.0)
    frames = []
    for branch in range(branches):
        seed_all(seed + branch)
        encoded = diffusion.sample_impute(
            subset[:, :train_data.d_numerical].to(diffusion.device),
            subset[:, train_data.d_numerical:].long().to(diffusion.device),
            positions, [], 3, "x_t", 0.0, 0.0,
        )
        frames.append(decode(train_data, encoded, info))
    values = np.stack([frame[output_names].apply(pd.to_numeric, errors="coerce").to_numpy(float) for frame in frames])
    scale = bank[output_names].apply(pd.to_numeric, errors="coerce").std().to_numpy(float)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    base_values = result.loc[indices, output_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    distance = np.nanmean(np.abs(values - base_values[None, :, :]) / scale[None, None, :], axis=2)
    valid = np.ones((branches, len(indices)), dtype=bool)
    shortfall = np.zeros((branches, len(indices)), dtype=float)
    for column, threshold in item.constraints:
        position = output_names.index(column)
        observed = values[:, :, position]
        valid &= observed >= threshold
        shortfall += np.maximum(threshold - observed, 0) / scale[position]
    scored = np.where(valid, distance, np.inf)
    choice = scored.argmin(axis=0)
    no_valid = ~valid.any(axis=0)
    if no_valid.any():
        choice[no_valid] = shortfall[:, no_valid].argmin(axis=0)
    selected = values[choice, np.arange(len(indices))]
    for position, column in enumerate(output_names):
        result.loc[indices, column] = selected[:, position]
    conditioned.set_condition(None, 1.0)
    return certify(result, item, violated), float(valid.any(axis=0).mean())


def empirical_joint(base: pd.DataFrame, bank: pd.DataFrame, item: Conjunction,
                    active: np.ndarray, violated: np.ndarray, seed: int):
    bank_a, bank_g, _ = masks(bank, item.context)
    reference = bank.loc[bank_a & bank_g & validity(bank, item)]
    result = base.copy(); indices = np.flatnonzero(violated)
    if len(indices) and len(reference):
        rng = np.random.default_rng(seed)
        chosen = reference.iloc[rng.choice(len(reference), len(indices), replace=True)]
        for column, _ in item.constraints:
            result.loc[indices, column] = pd.to_numeric(chosen[column]).to_numpy(float)
    return certify(result, item, active), len(reference)


def evaluate(sample: pd.DataFrame, base: pd.DataFrame, heldout: pd.DataFrame, info: dict,
             item: Conjunction, method: str, branch_success: float = math.nan,
             bank_support: int = 0) -> dict:
    base_a, base_g, _ = masks(base, item.context); active = base_a & base_g
    sample_a, sample_g, _ = masks(sample, item.context)
    real_a, real_g, _ = masks(heldout, item.context); real_active = real_a & real_g
    valid = validity(sample, item)
    numeric = [info["column_names"][index] for index in info["num_col_idx"]]
    categorical = [info["column_names"][index] for index in info["cat_col_idx"]]
    drift = []
    for column in numeric:
        scale = max(float(pd.to_numeric(heldout[column], errors="coerce").std()), 1e-8)
        if active.any():
            drift.append(float(np.nanmean(np.abs(pd.to_numeric(sample.loc[active, column]).to_numpy(float) -
                                                  pd.to_numeric(base.loc[active, column]).to_numpy(float))) / scale))
    target_ks = []
    for column, _ in item.constraints:
        if active.any() and real_active.any():
            target_ks.append(float(ks_2samp(pd.to_numeric(sample.loc[active, column]),
                                           pd.to_numeric(heldout.loc[real_active, column])).statistic))
    cohort_ks = [float(ks_2samp(pd.to_numeric(sample.loc[active, column]),
                                pd.to_numeric(heldout.loc[real_active, column])).statistic)
                 for column in numeric if active.any() and real_active.any()]
    cat_tv = [total_variation(sample.loc[active, column], heldout.loc[real_active, column])
              for column in categorical if active.any() and real_active.any()]
    return {
        "policy": item.name, "method": method, "active_n": int(active.sum()),
        "active_satisfaction": float(valid[active].mean()) if active.any() else math.nan,
        "active_numeric_drift_sd": float(np.mean(drift)),
        "target_ks": float(np.mean(target_ks)), "cohort_numeric_ks": float(np.mean(cohort_ks)),
        "cohort_categorical_tv": float(np.mean(cat_tv)),
        "antecedent_flip": float((sample_a != base_a).mean()), "subgroup_flip": float((sample_g != base_g).mean()),
        "bestof_branch_success": branch_success, "empirical_joint_support": bank_support,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--dataset", choices=["default", "shoppers"], required=True)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1536); parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260930); parser.add_argument("--guidance", type=float, default=2.0)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    started = time.perf_counter(); device = torch.device("cuda"); sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, args.dataset)
    bank = decode(train_data, train_data.X, info); heldout = decode(test_data, test_data.X, info)
    saved = torch.load(args.adapter, map_location="cpu", weights_only=False)
    with args.config.open("rb") as handle: config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"]); params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    backbone = ConditionedUniModMLP(condition_dim=int(saved["condition_dim"]), **params)
    base = Model(backbone, **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = base.load_state_dict(checkpoint["denoise_fn"], strict=False)
    missing = [key for key in incompatible.missing_keys if "condition_projection" not in key]
    if incompatible.unexpected_keys or missing: raise RuntimeError(str(incompatible))
    backbone.mlp.condition_projection.load_state_dict(saved["condition_projection"]); backbone.mlp.mlp.load_state_dict(saved["latent_mlp"])
    conditioned = ContractConditionedModel(base, backbone, int(saved["condition_dim"]))
    from subgroup_projection_online import PreviewModel
    preview = PreviewModel(conditioned, train_data.categories)
    dp = copy.deepcopy(config["diffusion_params"]); dp.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    dp["sampler_params"] = copy.deepcopy(dp["sampler_params"]); dp["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, preview, None, **dp, device=device).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"]); diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"]); diffusion.eval()

    rows = []
    for ordinal, item in enumerate(conjunctions(args.dataset)):
        common_seed = args.seed + ordinal * 10000
        zero = torch.zeros((args.samples, int(saved["condition_dim"])), device=device)
        conditioned.set_condition(zero, 1.0); seed_all(common_seed)
        base_encoded, trace = sample_unconditional_trace(diffusion, conditioned, args.samples)
        base_frame = decode(train_data, base_encoded, info)
        a, g, _ = masks(base_frame, item.context); active = a & g
        violated = active & ~validity(base_frame, item)
        rows.append(evaluate(base_frame, base_frame, heldout, info, item, "unconditional"))
        vector = torch.tensor(structured_vector({"predicates": [(c, "ge", t) for c, t in item.constraints]}, args.dataset), dtype=torch.float32)
        scope = numeric_scope(args.dataset, info, item)
        conditioned.set_numeric_output_mask([original_numeric_position(info, name) for name in scope])
        seed_all(common_seed)
        generated = sample_conditioned(diffusion, preview, conditioned, train_data, info, item.context, vector,
                                       args.samples, "frozen", 1.0, args.guidance, violated, trace)
        generated = decode(train_data, generated, info)
        rows.append(evaluate(generated, base_frame, heldout, info, item, "lookahead_conjunction_before_certifier"))
        rows.append(evaluate(certify(generated, item, active), base_frame, heldout, info, item,
                             "lookahead_conjunction_plus_certifier"))
        rows.append(evaluate(certify(base_frame, item, active), base_frame, heldout, info, item, "clamp_certifier"))
        selected, success = bestof(diffusion, conditioned, train_data, info, base_encoded, base_frame, bank,
                                   item, violated, scope, common_seed + 777, 8)
        rows.append(evaluate(selected, base_frame, heldout, info, item, "bestof8_matched_scope_plus_certifier", success))
        empirical, support = empirical_joint(base_frame, bank, item, active, violated, common_seed + 1777)
        rows.append(evaluate(empirical, base_frame, heldout, info, item, "empirical_joint_repair", bank_support=support))
    metrics = pd.DataFrame(rows); metrics.to_csv(args.output / "policy_metrics.csv", index=False)
    summary = metrics.groupby("method").agg(
        policies=("policy", "size"), satisfaction=("active_satisfaction", "mean"),
        drift=("active_numeric_drift_sd", "mean"), target_ks=("target_ks", "mean"),
        numeric_ks=("cohort_numeric_ks", "mean"), categorical_tv=("cohort_categorical_tv", "mean"),
        branch_success=("bestof_branch_success", "mean"), empirical_support=("empirical_joint_support", "mean"),
        antecedent_flip=("antecedent_flip", "mean"), subgroup_flip=("subgroup_flip", "mean"),
    ).reset_index(); summary.to_csv(args.output / "summary.csv", index=False)
    lines = ["# Subgroup conjunction screen", "",
             "| Method | Validity | Numeric drift | Target KS | Cohort numeric KS | Categorical TV | Best-of valid branch |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.method} | {row.satisfaction:.4f} | {row.drift:.4f} | {row.target_ks:.4f} | {row.numeric_ks:.4f} | {row.categorical_tv:.4f} | {row.branch_success:.4f} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "status": "executed", "dataset": args.dataset, "seed": args.seed, "samples": args.samples,
        "constraints_per_policy": 2, "best_of_n": 8, "runtime_seconds": time.perf_counter() - started,
    }, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
