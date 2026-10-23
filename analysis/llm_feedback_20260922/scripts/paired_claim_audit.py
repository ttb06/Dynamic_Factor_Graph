#!/usr/bin/env python3
"""Paired bootstrap audits for the two current central empirical claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interval(values: np.ndarray, seed: int, draws: int = 20000) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": np.nan, "low": np.nan, "high": np.nan, "win_rate": np.nan}
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "n": len(values), "mean": float(values.mean()),
        "low": float(np.quantile(sampled, 0.025)), "high": float(np.quantile(sampled, 0.975)),
        "win_rate": float((values > 0).mean()),
    }


def probability_controls(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("prob_gate_g*_t*_seed*/control_metrics.csv"))
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    data = data[data["split"].str.startswith("heldout")]
    keys = ["seed", "guidance_scale", "contract_id"]
    base = data[data.schedule.eq("never")][keys + ["conditional_violation_rate", "inactive_target_mean_error"]]
    base = base.rename(columns={"conditional_violation_rate":"base_violation", "inactive_target_mean_error":"base_inactive"})
    rows = []
    for (guidance, gate_start, schedule), group in data[~data.schedule.eq("never")].groupby(
        ["guidance_scale", "gate_start_t", "schedule"]
    ):
        paired = group.merge(base, on=keys, how="inner")
        validity = paired.base_violation.to_numpy() - paired.conditional_violation_rate.to_numpy()
        preservation = paired.base_inactive.to_numpy() - paired.inactive_target_mean_error.to_numpy()
        for metric, values in (("violation_reduction", validity), ("inactive_error_reduction", preservation)):
            result = interval(values, seed=round(1000 * guidance + 100 * gate_start) + len(schedule))
            rows.append({"experiment":"row_active_set", "guidance":guidance, "gate_start":gate_start,
                         "method":schedule, "metric":metric, **result})
    return pd.DataFrame(rows)


def cross_condition(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("cross_condition_*_seed*/contract_metrics.csv"))
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    data = data[data.split.ne("train")]
    keys = ["dataset", "seed", "guidance_scale", "contract_id", "split"]
    identifier = data[data.representation.eq("request_id")][keys + ["satisfaction_rate", "cohort_numeric_ks", "cohort_categorical_tv"]]
    structured = data[data.representation.eq("structured")][keys + ["satisfaction_rate", "cohort_numeric_ks", "cohort_categorical_tv"]]
    paired = structured.merge(identifier, on=keys, suffixes=("_structured", "_id"))
    rows = []
    for (dataset, guidance, split), group in paired.groupby(["dataset", "guidance_scale", "split"]):
        effects = {
            "satisfaction_gain": group.satisfaction_rate_structured.to_numpy() - group.satisfaction_rate_id.to_numpy(),
            "numeric_ks_reduction": group.cohort_numeric_ks_id.to_numpy() - group.cohort_numeric_ks_structured.to_numpy(),
            "categorical_tv_reduction": group.cohort_categorical_tv_id.to_numpy() - group.cohort_categorical_tv_structured.to_numpy(),
        }
        for metric, values in effects.items():
            result = interval(values, seed=len(dataset) * 1000 + round(guidance * 10) + len(split))
            rows.append({"experiment":"structured_transfer", "dataset":dataset, "guidance":guidance,
                         "split":split, "method":"structured_minus_request_id", "metric":metric, **result})
    return pd.DataFrame(rows)


def cross_baselines(root: Path) -> pd.DataFrame:
    condition_paths = sorted(root.glob("cross_condition_*_structured_seed*/contract_metrics.csv"))
    baseline_paths = sorted(root.glob("cross_actuation_v2_*_seed*/contract_metrics.csv"))
    if not baseline_paths:
        baseline_paths = sorted(root.glob("cross_actuation_*_seed*/contract_metrics.csv"))
    condition = pd.concat([pd.read_csv(path) for path in condition_paths], ignore_index=True)
    baseline = pd.concat([pd.read_csv(path) for path in baseline_paths], ignore_index=True)
    condition = condition[condition.split.ne("train")]
    baseline = baseline[baseline.split.ne("train")]
    keys = ["dataset", "seed", "contract_id", "split"]
    rows = []
    comparisons = [(2.0, "best_of_2"), (2.0, "best_of_4"), (4.0, "best_of_4"), (2.0, "final_projection")]
    for guidance, baseline_name in comparisons:
        left = condition[condition.guidance_scale.eq(guidance)]
        right = baseline[baseline.method.eq(baseline_name)]
        paired = left.merge(right, on=keys, suffixes=("_condition", "_baseline"))
        for (dataset, split), group in paired.groupby(["dataset", "split"]):
            sat = group.satisfaction_rate_condition.to_numpy() - group.satisfaction_rate_baseline.to_numpy()
            ks = group.cohort_numeric_ks_baseline.to_numpy() - group.cohort_numeric_ks_condition.to_numpy()
            tv = group.cohort_categorical_tv_baseline.to_numpy() - group.cohort_categorical_tv_condition.to_numpy()
            pareto = ((sat >= 0) & (ks >= 0) & (tv >= 0) & ((sat > 0) | (ks > 0) | (tv > 0))).astype(float)
            for metric, values in {
                "satisfaction_gain": sat,
                "numeric_ks_reduction": ks,
                "categorical_tv_reduction": tv,
                "per_contract_pareto_win": pareto,
            }.items():
                result = interval(values, seed=len(dataset) * 1000 + round(guidance * 100) + len(baseline_name) + len(split))
                rows.append({
                    "experiment":"matched_actuation", "dataset":dataset, "guidance":guidance,
                    "split":split, "method":f"structured_vs_{baseline_name}", "metric":metric, **result,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    probability = probability_controls(args.run_root)
    cross = cross_condition(args.run_root)
    actuation = cross_baselines(args.run_root)
    all_results = pd.concat([probability, cross, actuation], ignore_index=True, sort=False)
    all_results.to_csv(args.output / "paired_effects.csv", index=False)

    focus = pd.concat([
        probability[
            probability.method.isin([
                "oracle_baseline_applicability", "global_then_hard_0.25", "soft0_then_hard_0.25",
                "shuffled_then_hard_0.25",
            ])
        ],
        cross[(cross.guidance.eq(2)) & cross.metric.isin(["satisfaction_gain", "numeric_ks_reduction", "categorical_tv_reduction"])],
        actuation[
            actuation.method.isin(["structured_vs_best_of_2", "structured_vs_best_of_4", "structured_vs_final_projection"])
            & actuation.guidance.eq(2)
        ],
    ], ignore_index=True, sort=False)
    lines = [
        "# Paired claim audit", "",
        "Positive effects mean improvement. Intervals are nonparametric paired bootstrap intervals over contract–seed observations; they are exploratory, not confirmatory multi-dataset tests.", "",
        "| experiment | dataset | CFG | gate | split | method | metric | n | mean | 95% interval | win rate |",
        "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.experiment} | {getattr(row, 'dataset', '') if pd.notna(getattr(row, 'dataset', np.nan)) else ''} | "
            f"{row.guidance:g} | {getattr(row, 'gate_start', '') if pd.notna(getattr(row, 'gate_start', np.nan)) else ''} | "
            f"{getattr(row, 'split', '') if pd.notna(getattr(row, 'split', np.nan)) else ''} | {row.method} | {row.metric} | "
            f"{row.n} | {row.mean:.4f} | [{row.low:.4f}, {row.high:.4f}] | {row.win_rate:.3f} |"
        )
    lines += ["", "## Guardrails", "", "- The row-active-set control uses controlled Adult contracts and a learned adapter; it does not establish natural semantic correctness.", "- Request-ID is intentionally zero for unseen contracts, so the structured comparison demonstrates parameter/composition transfer, not superiority over every possible encoder.", "- Contracts and seeds are not independent draws from a population of all semantic requests; the intervals quantify stability of this benchmark only."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({"status":"executed", "probability_effects":len(probability), "cross_transfer_effects":len(cross), "actuation_effects":len(actuation), "bootstrap_draws":20000}, indent=2))
    print(f"wrote {len(all_results)} paired effects to {args.output}")


if __name__ == "__main__":
    main()
