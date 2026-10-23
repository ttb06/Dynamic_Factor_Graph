#!/usr/bin/env python3
"""Paired-noise online test of subgroup-gated numeric projection.

This is a CTDF-style projection proxy, not a reproduction of CTDF.  It asks a
narrow mechanism question: when an implication is reliable only in a subgroup,
does gating the projection by a denoised preview preserve valid exceptions
better than enforcing the same edge globally?  The policies and frozen
evaluation scopes are defined without external API calls.
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
from scipy.stats import ks_2samp

from conditioned_tabdiff_adapter import decode, seed_all
from tabdiff_probe import preprocess_dataset


@dataclass(frozen=True)
class Policy:
    name: str
    antecedent: str
    subgroup: str
    target: str


POLICIES = [
    Policy("pay3_when_bill4_positive", "PAY_4", "BILL_AMT4", "PAY_AMT3"),
    Policy("pay4_when_bill5_positive", "PAY_5", "BILL_AMT5", "PAY_AMT4"),
    Policy("pay5_when_bill6_positive", "PAY_6", "BILL_AMT6", "PAY_AMT5"),
]


class PreviewModel(torch.nn.Module):
    """Expose the latest denoised numeric preview and categorical prediction."""

    def __init__(self, base, category_sizes):
        super().__init__()
        self.base = base
        self.category_sizes = [int(value) for value in category_sizes]
        self.numeric = None
        self.codes = None

    def predicted_codes(self, x_cat, logits):
        current_chunks = torch.split(x_cat, [size + 1 for size in self.category_sizes], dim=-1)
        logit_chunks = torch.split(logits, [size + 1 for size in self.category_sizes], dim=-1)
        codes = []
        for size, current, prediction in zip(self.category_sizes, current_chunks, logit_chunks):
            current_code = current.argmax(dim=-1)
            predicted = prediction[:, :size].argmax(dim=-1)
            codes.append(torch.where(current_code.eq(size), predicted, current_code))
        return torch.stack(codes, dim=1)

    def forward(self, x_num, x_cat, t, sigma=None):
        output = self.base(x_num, x_cat, t, sigma=sigma)
        self.numeric = output[0].detach()
        self.codes = self.predicted_codes(x_cat, output[1]).detach()
        return output


def original_numeric_position(info: dict, name: str) -> int:
    original = info["column_names"].index(name)
    return info["num_col_idx"].index(original)


def internal_categorical_position(info: dict, name: str) -> int:
    # Preprocessing places the target first, followed by feature categoricals
    # in the order recorded by cat_col_idx.
    original = info["column_names"].index(name)
    return 1 + info["cat_col_idx"].index(original)


def categorical_code(train_data, internal_position: int, value: str) -> int:
    categories = train_data.cat_transform.named_steps["ordinal"].categories_[internal_position]
    matches = np.flatnonzero(np.asarray(categories).astype(str) == str(value))
    if len(matches) != 1:
        raise ValueError(f"categorical value {value!r} not uniquely found at position {internal_position}")
    return int(matches[0])


def encoded_numeric_value(train_data, real: pd.DataFrame, info: dict, column: str, value: float) -> float:
    names = [info["column_names"][index] for index in info["num_col_idx"]]
    probe = real[names].median(numeric_only=True).to_numpy(dtype=float)[None, :]
    position = names.index(column)
    probe[0, position] = value
    return float(train_data.num_transform.transform(probe)[0, position])


@torch.no_grad()
def sample_gated(
    diffusion,
    preview: PreviewModel,
    num_samples: int,
    policy: Policy,
    info: dict,
    train_data,
    real: pd.DataFrame,
    mode: str,
    start_t: float,
    frozen_gate: np.ndarray | None = None,
) -> torch.Tensor:
    """Project target x_t after a step, using the latest x-hat-0 for gating."""
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num_cur = diffusion.num_schedule.total_noise(times)
    sigma_cat_cur = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num_cur); sigma_num_next[1:] = sigma_num_cur[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat_cur); sigma_cat_next[1:] = sigma_cat_cur[:-1]
    if diffusion.sampler_params["stochastic_sampler"] or diffusion.sampler_params["second_order_correction"]:
        raise ValueError("paired projection probe requires deterministic first-order sampling")
    numeric = torch.randn((num_samples, diffusion.num_numerical_features), device=device) * sigma_num_cur[-1]
    categorical = diffusion._sample_masked_prior(num_samples, len(diffusion.num_classes))

    antecedent_position = internal_categorical_position(info, policy.antecedent)
    antecedent_code = categorical_code(train_data, antecedent_position, "-1")
    subgroup_position = original_numeric_position(info, policy.subgroup)
    target_position = original_numeric_position(info, policy.target)
    subgroup_zero = encoded_numeric_value(train_data, real, info, policy.subgroup, 0.0)
    target_positive = encoded_numeric_value(train_data, real, info, policy.target, 1.0)
    frozen = None if frozen_gate is None else torch.as_tensor(frozen_gate, device=device, dtype=torch.bool)

    for index in reversed(range(diffusion.num_timesteps)):
        numeric, categorical, _ = diffusion.edm_update(
            numeric, categorical, index,
            times[index], times[index - 1] if index > 0 else None, times[index],
            sigma_num_cur[index], sigma_num_next[index], sigma_num_cur[index],
            sigma_cat_cur[index], sigma_cat_next[index], sigma_cat_cur[index],
        )
        if float(times[index]) > start_t:
            continue
        if mode == "frozen":
            gate = frozen
        else:
            antecedent = preview.codes[:, antecedent_position].eq(antecedent_code)
            if mode == "global":
                gate = antecedent
            elif mode == "subgroup":
                gate = antecedent & preview.numeric[:, subgroup_position].gt(subgroup_zero)
            else:
                raise ValueError(mode)
        numeric[gate, target_position].clamp_(min=target_positive)
    if not torch.all(categorical < diffusion.mask_index):
        raise RuntimeError("categorical mask remained at final step")
    return torch.cat([numeric, categorical], dim=1).cpu()


def policy_masks(frame: pd.DataFrame, policy: Policy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    antecedent = frame[policy.antecedent].astype(str).str.strip().eq("-1").to_numpy()
    subgroup = pd.to_numeric(frame[policy.subgroup], errors="coerce").gt(0).fillna(False).to_numpy()
    valid = pd.to_numeric(frame[policy.target], errors="coerce").gt(0).fillna(False).to_numpy()
    return antecedent, subgroup, valid


def repair(frame: pd.DataFrame, policy: Policy, gate: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    bad = gate & pd.to_numeric(result[policy.target], errors="coerce").le(0).fillna(True).to_numpy()
    result.loc[bad, policy.target] = 1.0
    return result


def safe_rate(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else math.nan


def distribution_distance(sample: pd.DataFrame, real: pd.DataFrame, info: dict) -> tuple[float, float]:
    numeric = [info["column_names"][index] for index in info["num_col_idx"]]
    categorical = [info["column_names"][index] for index in info["cat_col_idx"]]
    ks = [ks_2samp(pd.to_numeric(sample[c]), pd.to_numeric(real[c])).statistic for c in numeric]
    tv = []
    for column in categorical:
        left = sample[column].astype(str).value_counts(normalize=True)
        right = real[column].astype(str).value_counts(normalize=True)
        labels = left.index.union(right.index)
        tv.append(0.5 * float((left.reindex(labels, fill_value=0) - right.reindex(labels, fill_value=0)).abs().sum()))
    return float(np.mean(ks)), float(np.mean(tv))


def evaluate(sample: pd.DataFrame, base: pd.DataFrame, real: pd.DataFrame, info: dict,
             policy: Policy, method: str, start_t: float) -> dict:
    base_a, base_g, _ = policy_masks(base, policy)
    active = base_a & base_g
    exception = base_a & ~base_g
    a, g, valid = policy_masks(sample, policy)
    target = pd.to_numeric(sample[policy.target]).to_numpy(dtype=float)
    base_target = pd.to_numeric(base[policy.target]).to_numpy(dtype=float)
    scale = max(float(pd.to_numeric(real[policy.target]).std()), 1e-6)
    mean_ks, mean_tv = distribution_distance(sample, real, info)
    real_a, real_g, _ = policy_masks(real, policy)
    real_scope = real_a & real_g
    target_ks = ks_2samp(target[active], pd.to_numeric(real.loc[real_scope, policy.target]))[0] if active.any() and real_scope.any() else math.nan
    return {
        "policy": policy.name, "method": method, "start_t": start_t,
        "base_active_n": int(active.sum()), "base_exception_n": int(exception.sum()),
        "active_satisfaction": safe_rate(valid, active),
        "exception_satisfaction": safe_rate(valid, exception),
        "active_target_drift_sd": safe_rate(np.abs(target - base_target) / scale, active),
        "exception_target_drift_sd": safe_rate(np.abs(target - base_target) / scale, exception),
        "outside_antecedent_drift_sd": safe_rate(np.abs(target - base_target) / scale, ~base_a),
        "active_changed": safe_rate(~np.isclose(target, base_target), active),
        "exception_changed": safe_rate(~np.isclose(target, base_target), exception),
        "antecedent_flip": float((a != base_a).mean()), "subgroup_flip": float((g != base_g).mean()),
        "mean_numeric_ks_real": mean_ks, "mean_categorical_tv_real": mean_tv,
        "active_target_ks_real": float(target_ks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    started = time.perf_counter()
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    info = json.loads((args.repo / "data/default/info.json").read_text())
    train_data, _ = preprocess_dataset(args.repo, "default")
    real = decode(train_data, train_data.X, info)
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    base_model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_model.load_state_dict(checkpoint["denoise_fn"], strict=True)
    preview = PreviewModel(base_model, train_data.categories)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, preview, None,
                                      **diffusion_params, device=torch.device("cuda")).cuda()
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    diffusion.eval()

    rows = []
    samples_out = {}
    for ordinal, policy in enumerate(POLICIES):
        common_seed = args.seed + 10000 * ordinal
        seed_all(common_seed)
        base = decode(train_data, diffusion.sample(args.samples), info)
        base_a, base_g, _ = policy_masks(base, policy)
        frozen_active = base_a & base_g
        rows.append(evaluate(base, base, real, info, policy, "unconditional", math.nan))

        methods = [
            ("static_graph_preview_t050", "global", 0.50, None),
            ("dynamic_graph_preview_t050", "subgroup", 0.50, None),
            ("dynamic_graph_preview_t025", "subgroup", 0.25, None),
            ("oracle_frozen_subgroup_t050", "frozen", 0.50, frozen_active),
        ]
        for name, mode, start_t, gate in methods:
            seed_all(common_seed)
            encoded = sample_gated(diffusion, preview, args.samples, policy, info, train_data, real,
                                   mode=mode, start_t=start_t, frozen_gate=gate)
            sample = decode(train_data, encoded, info)
            rows.append(evaluate(sample, base, real, info, policy, name, start_t))
            samples_out[(policy.name, name)] = sample

        subgroup_repair = repair(base, policy, frozen_active)
        global_repair = repair(base, policy, base_a)
        rows.append(evaluate(subgroup_repair, base, real, info, policy, "subgroup_final_repair", 0.0))
        rows.append(evaluate(global_repair, base, real, info, policy, "static_final_repair", 0.0))
        dynamic = samples_out[(policy.name, "dynamic_graph_preview_t050")]
        dynamic_certified = repair(dynamic, policy, frozen_active)
        rows.append(evaluate(dynamic_certified, base, real, info, policy, "dynamic_t050_plus_certifier", 0.50))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "policy_metrics.csv", index=False)
    summary = metrics.groupby("method", dropna=False).agg(
        active_satisfaction=("active_satisfaction", "mean"),
        exception_drift_sd=("exception_target_drift_sd", "mean"),
        exception_changed=("exception_changed", "mean"),
        outside_drift_sd=("outside_antecedent_drift_sd", "mean"),
        numeric_ks=("mean_numeric_ks_real", "mean"),
        categorical_tv=("mean_categorical_tv_real", "mean"),
        active_target_ks=("active_target_ks_real", "mean"),
        antecedent_flip=("antecedent_flip", "mean"),
        subgroup_flip=("subgroup_flip", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    lines = [
        "# Online subgroup-gated projection", "",
        "CTDF-style numeric projection proxy with paired initial noise; not a CTDF reproduction.", "",
        "| Method | Active satisfaction | Exception drift (SD) | Exceptions changed | Outside drift (SD) | Numeric KS | Active-target KS | A flip | G flip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.active_satisfaction:.4f} | {row.exception_drift_sd:.4f} | "
            f"{row.exception_changed:.4f} | {row.outside_drift_sd:.4f} | {row.numeric_ks:.4f} | "
            f"{row.active_target_ks:.4f} | {row.antecedent_flip:.4f} | {row.subgroup_flip:.4f} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    metadata = {
        "status": "executed", "dataset": "default", "seed": args.seed,
        "samples_per_policy": args.samples, "steps": args.steps,
        "policies": [policy.__dict__ for policy in POLICIES],
        "label": "CTDF-style subgroup projection proxy; not CTDF reproduction",
        "scope": "three repeated payment/bill policies; one seed; frozen base scope for paired evaluation",
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
