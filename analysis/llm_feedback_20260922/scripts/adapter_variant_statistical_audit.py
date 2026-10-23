#!/usr/bin/env python3
"""Paired, seed-clustered audit of condition-adapter training variants."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "catalog", "seed", "contract_id"]


def load(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"adapter_variant_frontier_*_seed*/{name}"))
    if not paths:
        raise FileNotFoundError(name)
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def cluster_interval(frame: pd.DataFrame, column: str, rng: np.random.Generator, draws: int = 5000) -> tuple[float, float]:
    clusters = frame.groupby("seed")[column].mean().dropna().to_numpy(float)
    if len(clusters) == 0:
        return np.nan, np.nan
    sample = rng.choice(clusters, size=(draws, len(clusters)), replace=True).mean(axis=1)
    return tuple(np.quantile(sample, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="fixed_proj")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    metrics = load(args.input_root, "frontier_metrics.csv")
    budgets = load(args.input_root, "budget_curve.csv")
    accepted = metrics[metrics["method"].isin(["condition_g1_accepted", "condition_g2_accepted"])].copy()
    accepted["guidance"] = accepted["method"].str.extract(r"condition_g([12])", expand=False).astype(int)
    raw = budgets[(budgets["budget"] == 1) & budgets["method"].isin(["condition_g1", "condition_g2"])].copy()
    raw["guidance"] = raw["method"].str.extract(r"condition_g([12])", expand=False).astype(int)
    runs = accepted.merge(
        raw[KEY + ["run_label", "guidance", "success_with_budget"]],
        on=KEY + ["run_label", "guidance"], validate="one_to_one",
    )
    runs = runs.rename(columns={"success_with_budget": "success"})
    runs.to_csv(args.output / "all_paired_measures.csv", index=False)

    reference = runs[runs["run_label"] == args.reference].copy()
    alternatives = runs[runs["run_label"] != args.reference].copy()
    measures = ["success", "numeric_ks", "categorical_tv", "target_numeric_ks", "nontarget_numeric_ks"]
    available = [column for column in measures if column in runs]
    paired = alternatives.merge(
        reference[KEY + ["guidance"] + available],
        on=KEY + ["guidance"], suffixes=("", "_reference"), validate="many_to_one",
    )
    for column in available:
        paired[f"delta_{column}"] = paired[column] - paired[f"{column}_reference"]
    paired.to_csv(args.output / "paired_vs_fixed_projection.csv", index=False)

    groups = ["dataset", "catalog", "guidance", "run_label"]
    rng = np.random.default_rng(20260929)
    rows = []
    for values, block in paired.groupby(groups):
        row = dict(zip(groups, values))
        row["seeds"] = block["seed"].nunique()
        row["paired_contract_runs"] = len(block)
        for column in available:
            delta = f"delta_{column}"
            row[delta] = block[delta].mean()
            low, high = cluster_interval(block, delta, rng)
            row[f"{delta}_ci_low"] = low
            row[f"{delta}_ci_high"] = high
        row["strict_mean_dominance"] = bool(
            row["delta_success"] > 0
            and row["delta_numeric_ks"] < 0
            and row["delta_categorical_tv"] < 0
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary_vs_fixed_projection.csv", index=False)

    display = summary.sort_values(
        ["strict_mean_dominance", "delta_success", "delta_numeric_ks", "delta_categorical_tv"],
        ascending=[False, False, True, True],
    )
    lines = [
        "# Adapter variant paired statistical audit",
        "",
        "All deltas are relative to fixed-threshold, projection-only training. Confidence intervals resample seed-cluster means; contracts within one seed are not treated as independent replications.",
        "",
        "| Dataset | Catalog | g | Variant | Δ success | 95% CI | Δ KS | 95% CI | Δ TV | 95% CI | Mean dominance |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.catalog} | {row.guidance} | {row.run_label} | "
            f"{row.delta_success:+.4f} | [{row.delta_success_ci_low:+.4f}, {row.delta_success_ci_high:+.4f}] | "
            f"{row.delta_numeric_ks:+.4f} | [{row.delta_numeric_ks_ci_low:+.4f}, {row.delta_numeric_ks_ci_high:+.4f}] | "
            f"{row.delta_categorical_tv:+.4f} | [{row.delta_categorical_tv_ci_low:+.4f}, {row.delta_categorical_tv_ci_high:+.4f}] | "
            f"{row.strict_mean_dominance} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
