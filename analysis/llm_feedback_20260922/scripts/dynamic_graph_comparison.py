#!/usr/bin/env python3
"""Paired dynamic-graph versus fixed dependency-neighborhood guidance analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = [
    "dataset", "catalog", "split", "seed", "contract_id", "profile", "peak",
    "floor", "cutoff", "neighbor_k",
]
METRICS = [
    "numeric_ks", "categorical_tv", "target_numeric_ks", "nontarget_numeric_ks",
    "target_categorical_tv", "nontarget_categorical_tv",
]


def read_all(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"temporal_contract_guidance_*/{filename}"))
    if not paths:
        raise FileNotFoundError(filename)
    frames = [pd.read_csv(path).assign(source=str(path.parent)) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if "actuation_scope" not in result:
        result["actuation_scope"] = "all"
    if "graph_candidate_k" not in result:
        result["graph_candidate_k"] = np.nan
    constant = result["profile"].eq("constant")
    result.loc[constant, "floor"] = result.loc[constant, "peak"]
    return result


def cluster_ci(values: pd.DataFrame, column: str, rng: np.random.Generator) -> tuple[float, float]:
    clustered = values.groupby("seed")[column].mean().to_numpy(float)
    clustered = clustered[np.isfinite(clustered)]
    if not len(clustered):
        return np.nan, np.nan
    draws = rng.choice(clustered, size=(4000, len(clustered)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    metrics = read_all(args.input_root, "temporal_metrics.csv")
    budgets = read_all(args.input_root, "temporal_budget.csv")
    metrics = metrics[metrics["method"].str.endswith("_accepted")].copy()
    available = [column for column in METRICS if column in metrics]
    metrics = metrics[KEY + ["actuation_scope", "graph_candidate_k"] + available]
    budgets = budgets[KEY + ["actuation_scope", "graph_candidate_k", "single_trajectory_success",
                              "expected_denoiser_calls_per_valid", "dynamic_graph_summary"]]
    metrics = metrics.drop_duplicates(KEY + ["actuation_scope", "graph_candidate_k"], keep="last")
    budgets = budgets.drop_duplicates(KEY + ["actuation_scope", "graph_candidate_k"], keep="last")
    runs = metrics.merge(
        budgets, on=KEY + ["actuation_scope", "graph_candidate_k"], validate="one_to_one"
    )
    dynamic = runs[runs["actuation_scope"].eq("dynamic_graph")].copy()
    fixed = runs[runs["actuation_scope"].eq("target_neighbors")].copy()
    fixed = fixed.drop(columns=["graph_candidate_k", "dynamic_graph_summary"])
    paired = dynamic.merge(fixed, on=KEY, suffixes=("_dynamic", "_fixed"), validate="many_to_one")
    compared = available + ["single_trajectory_success", "expected_denoiser_calls_per_valid"]
    for column in compared:
        paired[f"delta_{column}"] = paired[f"{column}_dynamic"] - paired[f"{column}_fixed"]
    paired.to_csv(args.output / "paired_rows.csv", index=False)

    group = ["dataset", "catalog", "split", "profile", "peak", "floor", "cutoff",
             "neighbor_k", "graph_candidate_k"]
    rows = []
    rng = np.random.default_rng(20260929)
    for values, block in paired.groupby(group, dropna=False):
        row = dict(zip(group, values))
        row["paired_cases"] = len(block)
        row["seeds"] = block["seed"].nunique()
        for column in [f"delta_{name}" for name in compared]:
            finite = block[column].to_numpy(float)
            finite = finite[np.isfinite(finite)]
            row[column] = float(finite.mean()) if len(finite) else np.nan
            row[f"{column}_ci_low"], row[f"{column}_ci_high"] = cluster_ci(block, column, rng)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)

    lines = [
        "# Dynamic constraint graph versus fixed dependency graph",
        "",
        ("Every row is paired by dataset, condition, adapter seed, noise seed, schedule and active-edge "
         "budget. Negative ΔKS/ΔTV favors the dynamic graph; positive Δsuccess favors it."),
        "",
        "| Dataset | Catalog | Split | Schedule | Active/Pool | Seeds | Δ success | Δ calls/valid | Δ KS | Δ TV |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not summary.empty:
        summary = summary.sort_values(["dataset", "catalog", "split", "neighbor_k"])
        for row in summary.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.catalog} | {row.split} | {row.profile} p={row.peak:g} | "
                f"{row.neighbor_k:g}/{row.graph_candidate_k:g} | {row.seeds} | "
                f"{row.delta_single_trajectory_success:+.4f} | "
                f"{row.delta_expected_denoiser_calls_per_valid:+.2f} | "
                f"{getattr(row, 'delta_numeric_ks', float('nan')):+.4f} | "
                f"{getattr(row, 'delta_categorical_tv', float('nan')):+.4f} |"
            )
    else:
        lines.append("No exact dynamic/fixed pairs were found.")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
