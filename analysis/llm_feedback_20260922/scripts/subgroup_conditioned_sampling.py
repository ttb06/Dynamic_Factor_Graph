#!/usr/bin/env python3
"""Row-wise dynamic use of an atomic condition adapter during reverse diffusion.

The adapter sees only atomic global threshold requests during training.  At
sampling time a preview-derived active set decides which rows receive that
condition.  This tests zero-shot composition of a condition foundation layer
with a dynamic subgroup graph.
"""
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
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

from conditioned_tabdiff_adapter import ConditionedUniModMLP, ContractConditionedModel, decode, seed_all
from mixed_type_condition_adapter import structured_vector
from subgroup_projection_online import PreviewModel, categorical_code, internal_categorical_position, original_numeric_position
from tabdiff_probe import preprocess_dataset


@dataclass(frozen=True)
class Policy:
    name: str
    antecedent: str
    antecedent_value: str
    subgroup: str
    target: str
    threshold: float
    regime: str
    subgroup_value: str | None = None


def policies(dataset: str, grid: bool = False) -> list[Policy]:
    if dataset == "shoppers":
        bases = [
            ("returning_weekend_pagevalues", "VisitorType", "Returning_Visitor", "Weekend", "PageValues", "True", [1, 5, 10, 25, 40]),
            ("november_returning_product", "Month", "Nov", "VisitorType", "ProductRelated_Duration", "Returning_Visitor", [200, 400, 600, 1500, 2800]),
            ("may_weekend_admin", "Month", "May", "Weekend", "Administrative_Duration", "True", [10, 50, 90, 150, 220]),
        ]
        if not grid:
            selected = {"returning_weekend_pagevalues": [5], "november_returning_product": [1500], "may_weekend_admin": [150]}
        else:
            selected = {name: thresholds for name, *_, thresholds in bases}
        return [Policy(f"{name}_{threshold:g}", antecedent, value, subgroup, target, float(threshold),
                       "controlled_request", subgroup_value)
                for name, antecedent, value, subgroup, target, subgroup_value, _ in bases
                for threshold in selected[name]]
    if dataset != "default": raise ValueError(dataset)
    bases = [
        ("pay2_bill3", "PAY_2", "0", "BILL_AMT3", "PAY_AMT3"),
        ("pay5_bill5", "PAY_5", "-1", "BILL_AMT5", "PAY_AMT4"),
        ("pay6_bill6", "PAY_6", "-1", "BILL_AMT6", "PAY_AMT5"),
    ]
    result = []
    thresholds = [1.0, 500.0, 1000.0, 2000.0, 5000.0] if grid else [1.0, 1000.0]
    for name, antecedent, value, subgroup, target in bases:
        for threshold in thresholds:
            regime = "empirical_positive" if threshold == 1 else "controlled_request"
            result.append(Policy(f"{name}_{threshold:g}", antecedent, value, subgroup, target, threshold, regime))
    return result


def masks(frame: pd.DataFrame, policy: Policy):
    antecedent = frame[policy.antecedent].astype(str).str.strip().eq(policy.antecedent_value).to_numpy()
    if policy.subgroup_value is None:
        subgroup = pd.to_numeric(frame[policy.subgroup], errors="coerce").gt(0).fillna(False).to_numpy()
    else:
        subgroup = frame[policy.subgroup].astype(str).str.strip().eq(policy.subgroup_value).to_numpy()
    valid = pd.to_numeric(frame[policy.target], errors="coerce").ge(policy.threshold).fillna(False).to_numpy()
    return antecedent, subgroup, valid


def ranked_numeric_descendants(bank: pd.DataFrame, info: dict, policy: Policy) -> list[str]:
    """Rank train-only numeric descendants by within-applicability association.

    The target is always first.  Ranking is deliberately deterministic and
    excludes the numeric subgroup key, which is anchored during rollback.
    """
    numeric_names = [info["column_names"][index] for index in info["num_col_idx"]]
    if policy.subgroup_value is None:
        numeric_names = [name for name in numeric_names if name != policy.subgroup]
    if info.get("name") == "default" or policy.target.startswith("PAY_AMT"):
        numeric_names = [name for name in numeric_names
                         if name.startswith("BILL_AMT") or name.startswith("PAY_AMT")]
    antecedent, subgroup, _ = masks(bank, policy)
    cohort = bank.loc[antecedent & subgroup]
    target = pd.to_numeric(cohort[policy.target], errors="coerce")
    scores = []
    for name in numeric_names:
        if name == policy.target:
            continue
        values = pd.to_numeric(cohort[name], errors="coerce")
        score = target.corr(values, method="spearman")
        scores.append((float(abs(score)) if pd.notna(score) else -1.0, name))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [policy.target] + [name for _, name in scores]


@torch.no_grad()
def sample_unconditional_trace(diffusion, conditioned: ContractConditionedModel, num_samples: int):
    """Sample once and retain the reverse trajectory for typed context anchoring."""
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num_cur = diffusion.num_schedule.total_noise(times)
    sigma_cat_cur = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num_cur); sigma_num_next[1:] = sigma_num_cur[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat_cur); sigma_cat_next[1:] = sigma_cat_cur[:-1]
    numeric = torch.randn((num_samples, diffusion.num_numerical_features), device=device) * sigma_num_cur[-1]
    categorical = diffusion._sample_masked_prior(num_samples, len(diffusion.num_classes))
    conditioned.set_condition(torch.zeros((num_samples, conditioned.condition_dim), device=device), 1.0)
    trace = []
    for index in reversed(range(diffusion.num_timesteps)):
        numeric, categorical, _ = diffusion.edm_update(
            numeric, categorical, index,
            times[index], times[index - 1] if index > 0 else None, times[index],
            sigma_num_cur[index], sigma_num_next[index], sigma_num_cur[index],
            sigma_cat_cur[index], sigma_cat_next[index], sigma_cat_cur[index],
        )
        trace.append((numeric.detach().clone(), categorical.detach().clone()))
    conditioned.set_condition(None, 1.0)
    return torch.cat([numeric, categorical], dim=1).cpu(), trace


@torch.no_grad()
def sample_conditioned(diffusion, preview: PreviewModel, conditioned: ContractConditionedModel,
                       train_data, info: dict, policy: Policy, vector: torch.Tensor,
                       num_samples: int, mode: str, start_t: float, guidance: float,
                       frozen_gate: np.ndarray | None = None, anchor_trace=None) -> torch.Tensor:
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num_cur = diffusion.num_schedule.total_noise(times)
    sigma_cat_cur = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num_cur); sigma_num_next[1:] = sigma_num_cur[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat_cur); sigma_cat_next[1:] = sigma_cat_cur[:-1]
    numeric = torch.randn((num_samples, diffusion.num_numerical_features), device=device) * sigma_num_cur[-1]
    categorical = diffusion._sample_masked_prior(num_samples, len(diffusion.num_classes))
    zero = torch.zeros((num_samples, len(vector)), device=device)
    conditioned.set_condition(zero, guidance)
    antecedent_position = internal_categorical_position(info, policy.antecedent)
    antecedent_code = categorical_code(train_data, antecedent_position, policy.antecedent_value)
    # Quantile transform is monotonic, so >0 can be tested against the encoded
    # zero obtained from the fitted train transform.
    if policy.subgroup_value is None:
        subgroup_position = original_numeric_position(info, policy.subgroup)
        probe = np.median(train_data.num_inverse(train_data.X[:, :train_data.d_numerical].numpy()), axis=0, keepdims=True)
        probe[0, subgroup_position] = 0.0
        subgroup_zero = float(train_data.num_transform.transform(probe)[0, subgroup_position])
        subgroup_cat_position = subgroup_code = None
    else:
        subgroup_position = subgroup_zero = None
        subgroup_cat_position = internal_categorical_position(info, policy.subgroup)
        subgroup_code = categorical_code(train_data, subgroup_cat_position, policy.subgroup_value)
    frozen = None if frozen_gate is None else torch.as_tensor(frozen_gate, dtype=torch.bool, device=device)
    active_vector = vector.to(device).repeat(num_samples, 1)
    for trace_index, index in enumerate(reversed(range(diffusion.num_timesteps))):
        numeric, categorical, _ = diffusion.edm_update(
            numeric, categorical, index,
            times[index], times[index - 1] if index > 0 else None, times[index],
            sigma_num_cur[index], sigma_num_next[index], sigma_num_cur[index],
            sigma_cat_cur[index], sigma_cat_next[index], sigma_cat_cur[index],
        )
        if anchor_trace is not None:
            anchor_numeric, anchor_categorical = anchor_trace[trace_index]
            if subgroup_position is not None:
                numeric[:, subgroup_position] = anchor_numeric[:, subgroup_position]
            else:
                categorical[:, subgroup_cat_position] = anchor_categorical[:, subgroup_cat_position]
            categorical[:, antecedent_position] = anchor_categorical[:, antecedent_position]
        gate = torch.zeros(num_samples, dtype=torch.bool, device=device)
        if float(times[index]) <= start_t:
            if mode == "frozen":
                gate = frozen
            else:
                antecedent = preview.codes[:, antecedent_position].eq(antecedent_code)
                if mode == "static":
                    gate = antecedent
                elif mode == "dynamic":
                    subgroup = (preview.numeric[:, subgroup_position].gt(subgroup_zero) if subgroup_position is not None
                                else preview.codes[:, subgroup_cat_position].eq(subgroup_code))
                    gate = antecedent & subgroup
                else:
                    raise ValueError(mode)
        conditioned.set_condition(torch.where(gate[:, None], active_vector, zero), guidance)
    conditioned.set_condition(None, 1.0)
    return torch.cat([numeric, categorical], dim=1).cpu()


def repair(frame: pd.DataFrame, policy: Policy, gate: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    bad = gate & pd.to_numeric(result[policy.target], errors="coerce").lt(policy.threshold).fillna(True).to_numpy()
    result.loc[bad, policy.target] = policy.threshold
    return result


def empirical_repair(frame: pd.DataFrame, bank: pd.DataFrame, policy: Policy, gate: np.ndarray,
                     rng: np.random.Generator, replace_all: bool, stratified: bool) -> pd.DataFrame:
    """Strong data-assisted postrepair using train rows only.

    This is intentionally stronger than point clamping and must be labelled as
    a train-bank baseline rather than an LCATS/Zhao reproduction.
    """
    result = frame.copy()
    bank_a, bank_g, bank_valid = masks(bank, policy)
    reference = bank.loc[bank_a & bank_g & bank_valid].copy()
    if reference.empty: raise RuntimeError(f"empty empirical repair bank for {policy.name}")
    current = pd.to_numeric(result[policy.target], errors="coerce")
    selected = gate if replace_all else gate & current.lt(policy.threshold).fillna(True).to_numpy()
    indices = np.flatnonzero(selected)
    if not len(indices): return result
    if not stratified:
        values = pd.to_numeric(reference[policy.target]).to_numpy(float)
        result.loc[selected, policy.target] = rng.choice(values, size=len(indices), replace=True)
        return result
    # Match on the subgroup magnitude using train-bank quantile bins.  This
    # preserves one important target-context dependency without copying rows.
    if policy.subgroup_value is not None:
        values = pd.to_numeric(reference[policy.target]).to_numpy(float)
        result.loc[selected, policy.target] = rng.choice(values, size=len(indices), replace=True)
        return result
    bank_context = pd.to_numeric(reference[policy.subgroup], errors="coerce").to_numpy(float)
    sample_context = pd.to_numeric(result.loc[selected, policy.subgroup], errors="coerce").to_numpy(float)
    cuts = np.unique(np.quantile(bank_context, np.linspace(0, 1, 11)))
    if len(cuts) < 3:
        values = pd.to_numeric(reference[policy.target]).to_numpy(float)
        result.loc[selected, policy.target] = rng.choice(values, size=len(indices), replace=True)
        return result
    bank_bins = np.clip(np.searchsorted(cuts, bank_context, side="right") - 1, 0, len(cuts) - 2)
    sample_bins = np.clip(np.searchsorted(cuts, sample_context, side="right") - 1, 0, len(cuts) - 2)
    replacement = np.empty(len(indices), dtype=float)
    bank_target = pd.to_numeric(reference[policy.target]).to_numpy(float)
    for position, bin_id in enumerate(sample_bins):
        candidates = bank_target[bank_bins == bin_id]
        if not len(candidates): candidates = bank_target
        replacement[position] = rng.choice(candidates)
    result.loc[selected, policy.target] = replacement
    return result


@torch.no_grad()
def diffusion_impute_repair(diffusion, conditioned: ContractConditionedModel, train_data, info: dict,
                            base_encoded: torch.Tensor, base_frame: pd.DataFrame, policy: Policy,
                            violated: np.ndarray, seed: int, rounds: int = 3,
                            scope: str = "target") -> pd.DataFrame:
    """Post-generation diffusion imputation under target or matched graph scope."""
    result = base_frame.copy()
    indices = np.flatnonzero(violated)
    if not len(indices): return result
    subset = base_encoded[indices]
    target_position = original_numeric_position(info, policy.target)
    if scope == "target":
        output_names = [policy.target]
    elif scope == "numgraph":
        if policy.subgroup_value is None:
            output_names = [name for name in info["column_names"]
                            if (name.startswith("BILL_AMT") or name.startswith("PAY_AMT"))
                            and name != policy.subgroup]
        else:
            output_names = [info["column_names"][index] for index in info["num_col_idx"]]
    else:
        raise ValueError(scope)
    output_positions = [original_numeric_position(info, name) for name in output_names]
    conditioned.set_numeric_output_mask(None)
    conditioned.set_condition(torch.zeros((len(indices), conditioned.condition_dim), device=diffusion.device), 1.0)
    seed_all(seed)
    imputed = diffusion.sample_impute(
        subset[:, :train_data.d_numerical].to(diffusion.device),
        subset[:, train_data.d_numerical:].long().to(diffusion.device),
        output_positions, [], rounds, "x_t", 0.0, 0.0,
    )
    decoded = decode(train_data, imputed, info)
    for name in output_names:
        result.loc[indices, name] = pd.to_numeric(decoded[name]).to_numpy(float)
    conditioned.set_condition(None, 1.0)
    return repair(result, policy, violated)


@torch.no_grad()
def diffusion_bestof_repair(diffusion, conditioned: ContractConditionedModel, train_data, info: dict,
                            base_encoded: torch.Tensor, base_frame: pd.DataFrame, bank: pd.DataFrame,
                            policy: Policy, violated: np.ndarray, seed: int, candidates: int,
                            scope: str) -> pd.DataFrame:
    """Select the closest valid branch from repeated unconditional imputations."""
    result = base_frame.copy()
    indices = np.flatnonzero(violated)
    if not len(indices):
        return result
    if scope == "target":
        output_names = [policy.target]
    elif scope == "numgraph":
        if policy.subgroup_value is None:
            output_names = [name for name in info["column_names"]
                            if (name.startswith("BILL_AMT") or name.startswith("PAY_AMT"))
                            and name != policy.subgroup]
        else:
            output_names = [info["column_names"][index] for index in info["num_col_idx"]]
    else:
        raise ValueError(scope)
    output_positions = [original_numeric_position(info, name) for name in output_names]
    subset = base_encoded[indices]
    conditioned.set_numeric_output_mask(None)
    conditioned.set_condition(torch.zeros((len(indices), conditioned.condition_dim), device=diffusion.device), 1.0)
    decoded_candidates = []
    for branch in range(candidates):
        seed_all(seed + branch)
        imputed = diffusion.sample_impute(
            subset[:, :train_data.d_numerical].to(diffusion.device),
            subset[:, train_data.d_numerical:].long().to(diffusion.device),
            output_positions, [], 3, "x_t", 0.0, 0.0,
        )
        decoded = decode(train_data, imputed, info)
        decoded_candidates.append(decoded[output_names].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    candidate_values = np.stack(decoded_candidates, axis=0)
    base_values = result.loc[indices, output_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    scale = bank[output_names].apply(pd.to_numeric, errors="coerce").std().to_numpy(float)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    distance = np.nanmean(np.abs(candidate_values - base_values[None, :, :]) / scale[None, None, :], axis=2)
    target_index = output_names.index(policy.target)
    valid = candidate_values[:, :, target_index] >= policy.threshold
    valid_distance = np.where(valid, distance, np.inf)
    choice = valid_distance.argmin(axis=0)
    no_valid = ~valid.any(axis=0)
    if no_valid.any():
        target_values = np.nan_to_num(candidate_values[:, :, target_index], nan=-np.inf)
        choice[no_valid] = target_values[:, no_valid].argmax(axis=0)
    selected = candidate_values[choice, np.arange(len(indices)), :]
    for column, name in enumerate(output_names):
        result.loc[indices, name] = selected[:, column]
    conditioned.set_condition(None, 1.0)
    return repair(result, policy, violated)


def rate(values: np.ndarray, selected: np.ndarray) -> float:
    return float(values[selected].mean()) if selected.any() else math.nan


def evaluate(sample: pd.DataFrame, base: pd.DataFrame, real: pd.DataFrame, policy: Policy, method: str, guidance: float) -> dict:
    base_a, base_g, _ = masks(base, policy)
    active, exception = base_a & base_g, base_a & ~base_g
    a, g, valid = masks(sample, policy)
    target = pd.to_numeric(sample[policy.target]).to_numpy(float)
    base_target = pd.to_numeric(base[policy.target]).to_numpy(float)
    scale = max(float(pd.to_numeric(real[policy.target]).std()), 1e-6)
    real_a, real_g, real_valid = masks(real, policy)
    reference_frame = real.loc[real_a & real_g & real_valid]
    reference = pd.to_numeric(reference_frame[policy.target]).to_numpy(float)
    cohort = sample.loc[active]
    numeric_columns = [column for column in real.columns if pd.api.types.is_numeric_dtype(real[column])]
    categorical_columns = [column for column in real.columns if column not in numeric_columns]
    numeric_ks = [ks_2samp(pd.to_numeric(cohort[column]), pd.to_numeric(reference_frame[column])).statistic
                  for column in numeric_columns if len(cohort) and len(reference_frame)]
    categorical_tv = []
    for column in categorical_columns:
        left = cohort[column].astype(str).value_counts(normalize=True)
        right = reference_frame[column].astype(str).value_counts(normalize=True)
        labels = left.index.union(right.index)
        categorical_tv.append(0.5 * float((left.reindex(labels, fill_value=0) - right.reindex(labels, fill_value=0)).abs().sum()))
    correlations = []
    if len(cohort) > 2 and len(reference_frame) > 2:
        for column in numeric_columns:
            if column == policy.target: continue
            sample_corr = spearmanr(pd.to_numeric(cohort[policy.target]), pd.to_numeric(cohort[column]), nan_policy="omit").statistic
            real_corr = spearmanr(pd.to_numeric(reference_frame[policy.target]), pd.to_numeric(reference_frame[column]), nan_policy="omit").statistic
            if np.isfinite(sample_corr) and np.isfinite(real_corr): correlations.append(abs(sample_corr - real_corr))
    target_scale = max(float(pd.to_numeric(real[policy.target]).std()), 1e-6)
    return {
        "policy": policy.name, "regime": policy.regime, "threshold": policy.threshold,
        "method": method, "guidance": guidance, "active_n": int(active.sum()), "exception_n": int(exception.sum()),
        "active_satisfaction": rate(valid, active), "exception_satisfaction": rate(valid, exception),
        "active_drift_sd": rate(np.abs(target - base_target) / scale, active),
        "exception_drift_sd": rate(np.abs(target - base_target) / scale, exception),
        "outside_drift_sd": rate(np.abs(target - base_target) / scale, ~base_a),
        "active_changed": rate(~np.isclose(target, base_target), active),
        "exception_changed": rate(~np.isclose(target, base_target), exception),
        "antecedent_flip": float((a != base_a).mean()), "subgroup_flip": float((g != base_g).mean()),
        "active_target_ks_reference": float(ks_2samp(target[active], reference).statistic) if active.any() and len(reference) else math.nan,
        "active_target_wasserstein_sd": float(wasserstein_distance(target[active], reference) / target_scale) if active.any() and len(reference) else math.nan,
        "active_cohort_numeric_ks": float(np.mean(numeric_ks)) if numeric_ks else math.nan,
        "active_cohort_categorical_tv": float(np.mean(categorical_tv)) if categorical_tv else math.nan,
        "target_context_spearman_mae": float(np.mean(correlations)) if correlations else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", choices=["default", "shoppers"], default="default")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--guidances", default="1:2")
    parser.add_argument("--policy-grid", action="store_true")
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--best-of-n", type=int, default=0)
    parser.add_argument("--heldout-thresholds-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    started = time.perf_counter(); device = torch.device("cuda")
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, args.dataset)
    repair_bank = decode(train_data, train_data.X, info)
    heldout_real = decode(test_data, test_data.X, info)
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    with args.config.open("rb") as handle: config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    backbone = ConditionedUniModMLP(condition_dim=int(adapter["condition_dim"]), **params)
    base = Model(backbone, **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = base.load_state_dict(checkpoint["denoise_fn"], strict=False)
    missing = [key for key in incompatible.missing_keys if "condition_projection" not in key]
    if incompatible.unexpected_keys or missing: raise RuntimeError(str(incompatible))
    backbone.mlp.condition_projection.load_state_dict(adapter["condition_projection"])
    backbone.mlp.mlp.load_state_dict(adapter["latent_mlp"])
    conditioned = ContractConditionedModel(base, backbone, int(adapter["condition_dim"]))
    preview = PreviewModel(conditioned, train_data.categories)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, preview, None,
                                      **diffusion_params, device=device).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    diffusion.eval()

    policy_list = policies(args.dataset, args.policy_grid)
    if args.heldout_thresholds_only:
        heldout = ({500.0, 2000.0} if args.dataset == "default"
                   else {5.0, 25.0, 50.0, 150.0, 1500.0})
        policy_list = [policy for policy in policy_list if policy.threshold in heldout]
        if not policy_list:
            raise RuntimeError("heldout threshold filter selected no policies")
    rows = []
    for ordinal, policy in enumerate(policy_list):
        common_seed = args.seed + 10000 * ordinal
        zero = torch.zeros((args.samples, int(adapter["condition_dim"])), device=device)
        conditioned.set_condition(zero, 1.0); seed_all(common_seed)
        base_encoded, base_trace = sample_unconditional_trace(diffusion, conditioned, args.samples)
        base_sample = decode(train_data, base_encoded, info)
        base_a, base_g, base_valid = masks(base_sample, policy)
        frozen = base_a & base_g
        frozen_violated = frozen & ~base_valid
        rows.append(evaluate(base_sample, base_sample, heldout_real, policy, "unconditional", 0.0))
        contract = {"predicates": [(policy.target, "ge", policy.threshold)]}
        vector = torch.tensor(structured_vector(contract, args.dataset), dtype=torch.float32)
        ranked_descendants = ranked_numeric_descendants(repair_bank, info, policy)
        for guidance in [float(value) for value in args.guidances.replace(",", ":").split(":") if value]:
            method_specs = [
                ("dynamic_graph_condition_t100", "dynamic", 1.0, None, None),
                ("dynamic_graph_condition_t075", "dynamic", 0.75, None, None),
                ("static_graph_condition_t050", "static", 0.5, None, None),
                ("dynamic_graph_condition_t050", "dynamic", 0.5, None, None),
                ("dynamic_graph_condition_t025", "dynamic", 0.25, None, None),
                ("oracle_frozen_condition_t100", "frozen", 1.0, frozen, None),
                ("oracle_frozen_condition_t075", "frozen", 0.75, frozen, None),
                ("oracle_frozen_condition_t050", "frozen", 0.5, frozen, None),
                ("lookahead_anchored_condition_t100", "frozen", 1.0, frozen, base_trace),
                ("lookahead_anchored_violated_t100", "frozen", 1.0, frozen_violated, base_trace),
                ("lookahead_anchored_violated_masked_t100", "frozen", 1.0, frozen_violated, base_trace),
                ("lookahead_anchored_violated_topk3_t100", "frozen", 1.0, frozen_violated, base_trace),
                ("lookahead_anchored_violated_topk5_t100", "frozen", 1.0, frozen_violated, base_trace),
                ("lookahead_anchored_violated_numgraph_t100", "frozen", 1.0, frozen_violated, base_trace),
            ]
            if args.focused:
                method_specs = [item for item in method_specs if item[0] in {
                    "lookahead_anchored_violated_t100",
                    "lookahead_anchored_violated_masked_t100",
                    "lookahead_anchored_violated_topk3_t100",
                    "lookahead_anchored_violated_topk5_t100",
                    "lookahead_anchored_violated_numgraph_t100",
                }]
            for method, mode, start_t, gate, anchor in method_specs:
                target_position = original_numeric_position(info, policy.target)
                if "_masked_" in method:
                    output_mask = [target_position]
                elif "_topk3_" in method:
                    output_mask = [original_numeric_position(info, name) for name in ranked_descendants[:3]]
                elif "_topk5_" in method:
                    output_mask = [original_numeric_position(info, name) for name in ranked_descendants[:5]]
                elif "_numgraph_" in method:
                    if args.dataset == "default":
                        output_names = [name for name in info["column_names"]
                                        if name.startswith("BILL_AMT") or name.startswith("PAY_AMT")]
                    else:
                        output_names = [info["column_names"][index] for index in info["num_col_idx"]]
                    output_mask = [original_numeric_position(info, name) for name in output_names]
                else:
                    output_mask = None
                conditioned.set_numeric_output_mask(output_mask)
                seed_all(common_seed)
                encoded = sample_conditioned(diffusion, preview, conditioned, train_data, info, policy, vector,
                                             args.samples, mode, start_t, guidance, gate, anchor)
                sample = decode(train_data, encoded, info)
                evaluated = evaluate(sample, base_sample, heldout_real, policy, method, guidance)
                evaluated["scope_columns"] = "|".join(
                    info["column_names"][info["num_col_idx"][position]] for position in output_mask
                ) if output_mask is not None else "all"
                rows.append(evaluated)
                if math.isclose(start_t, 1.0) and guidance == max(float(value) for value in args.guidances.replace(",", ":").split(":") if value) and mode in {"dynamic", "frozen"}:
                    certified = repair(sample, policy, frozen)
                    evaluated = evaluate(certified, base_sample, heldout_real, policy, method + "_plus_certifier", guidance)
                    evaluated["scope_columns"] = "|".join(
                        info["column_names"][info["num_col_idx"][position]] for position in output_mask
                    ) if output_mask is not None else "all"
                    rows.append(evaluated)
        conditioned.set_numeric_output_mask(None)
        rows.append(evaluate(repair(base_sample, policy, frozen), base_sample, heldout_real, policy, "subgroup_final_repair", 0.0))
        rows.append(evaluate(repair(base_sample, policy, base_a), base_sample, heldout_real, policy, "static_final_repair", 0.0))
        for name, replace_all, stratified in [
            ("empirical_marginal_bad_postrepair", False, False),
            ("empirical_stratified_bad_postrepair", False, True),
            ("empirical_marginal_all_postrepair", True, False),
        ]:
            repaired = empirical_repair(base_sample, repair_bank, policy, frozen,
                                        np.random.default_rng(common_seed + len(name)), replace_all, stratified)
            rows.append(evaluate(repaired, base_sample, heldout_real, policy, name, 0.0))
        imputed = diffusion_impute_repair(diffusion, conditioned, train_data, info, base_encoded, base_sample,
                                          policy, frozen_violated, common_seed + 777, rounds=3)
        rows.append(evaluate(imputed, base_sample, heldout_real, policy, "diffusion_impute_then_certify", 0.0))
        imputed_graph = diffusion_impute_repair(diffusion, conditioned, train_data, info, base_encoded, base_sample,
                                                policy, frozen_violated, common_seed + 1777, rounds=3,
                                                scope="numgraph")
        rows.append(evaluate(imputed_graph, base_sample, heldout_real, policy,
                             "diffusion_impute_numgraph_then_certify", 0.0))
        if args.best_of_n > 0:
            for scope in ["target", "numgraph"]:
                best = diffusion_bestof_repair(
                    diffusion, conditioned, train_data, info, base_encoded, base_sample, repair_bank,
                    policy, frozen_violated, common_seed + (2777 if scope == "target" else 3777),
                    args.best_of_n, scope,
                )
                rows.append(evaluate(
                    best, base_sample, heldout_real, policy,
                    f"diffusion_bestof{args.best_of_n}_{scope}_then_certify", 0.0,
                ))
    metrics = pd.DataFrame(rows); metrics.to_csv(args.output / "policy_metrics.csv", index=False)
    summary = metrics.groupby(["regime", "method", "guidance"], dropna=False).agg(
        active_satisfaction=("active_satisfaction", "mean"), exception_drift_sd=("exception_drift_sd", "mean"),
        exception_changed=("exception_changed", "mean"), outside_drift_sd=("outside_drift_sd", "mean"),
        active_drift_sd=("active_drift_sd", "mean"), active_changed=("active_changed", "mean"),
        target_ks=("active_target_ks_reference", "mean"), antecedent_flip=("antecedent_flip", "mean"),
        target_wasserstein=("active_target_wasserstein_sd", "mean"),
        cohort_numeric_ks=("active_cohort_numeric_ks", "mean"),
        cohort_categorical_tv=("active_cohort_categorical_tv", "mean"),
        target_context_corr_mae=("target_context_spearman_mae", "mean"),
        subgroup_flip=("subgroup_flip", "mean"),
    ).reset_index(); summary.to_csv(args.output / "summary.csv", index=False)
    lines = ["# Dynamic subgroup graph with atomic condition adapter", "",
             "The adapter was trained on atomic global thresholds only; row-wise subgroup activation is zero-shot composition.", "",
             "| Regime | Method | CFG | Satisfaction | Exception drift | Exceptions changed | Outside drift | Target KS | A flip | G flip |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.regime} | {row.method} | {row.guidance:.1f} | {row.active_satisfaction:.4f} | {row.exception_drift_sd:.4f} | {row.exception_changed:.4f} | {row.outside_drift_sd:.4f} | {row.target_ks:.4f} | {row.antecedent_flip:.4f} | {row.subgroup_flip:.4f} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "status": "executed", "dataset": args.dataset, "seed": args.seed, "samples_per_policy": args.samples,
        "steps": args.steps, "adapter": str(args.adapter), "atomic_training_only": True,
        "best_of_n": args.best_of_n,
        "heldout_thresholds_only": args.heldout_thresholds_only,
        "rowwise_subgroup_composition_seen_in_training": False, "runtime_seconds": time.perf_counter() - started,
    }, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__": main()
