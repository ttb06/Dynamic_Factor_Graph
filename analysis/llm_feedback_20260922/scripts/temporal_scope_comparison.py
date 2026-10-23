#!/usr/bin/env python3
"""Paired comparison of all-, target-, and all-to-target guidance scopes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["dataset", "catalog", "split", "seed", "contract_id", "profile", "peak", "floor", "cutoff"]
METRICS = [
    "numeric_ks", "categorical_tv", "target_numeric_ks", "nontarget_numeric_ks",
    "target_categorical_tv", "nontarget_categorical_tv",
]


def read_all(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"temporal_contract_guidance_*/{filename}"))
    if not paths:
        raise FileNotFoundError(filename)
    return pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in paths], ignore_index=True)


def scope_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # ``floor`` is not used by the constant schedule.  Normalize it so runs
    # submitted with f=0 and f=peak remain exact experimental matches.
    constant = result["profile"].eq("constant")
    result.loc[constant, "floor"] = result.loc[constant, "peak"]
    if "actuation_scope" not in result:
        result["actuation_scope"] = np.select(
            [
                result["method"].str.contains("_all2target_sc"),
                result["method"].str.contains("_neighbors"),
                result["method"].str.contains("_target"),
            ],
            ["all_then_target", "target_neighbors", "target"],
            default="all",
        )
    result["actuation_scope"] = result["actuation_scope"].fillna("all")
    if "scope_cutoff" not in result:
        result["scope_cutoff"] = np.nan
    result.loc[result["actuation_scope"] != "all_then_target", "scope_cutoff"] = np.nan
    if "neighbor_k" not in result:
        result["neighbor_k"] = np.nan
    result.loc[~result["actuation_scope"].isin(["target_neighbors", "dynamic_graph"]), "neighbor_k"] = np.nan
    if "graph_candidate_k" not in result:
        result["graph_candidate_k"] = np.nan
    result.loc[result["actuation_scope"] != "dynamic_graph", "graph_candidate_k"] = np.nan
    return result


def bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int = 4000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    metrics = scope_columns(read_all(args.input_root, "temporal_metrics.csv"))
    budgets = scope_columns(read_all(args.input_root, "temporal_budget.csv"))
    accepted = metrics[metrics["method"].str.endswith("_accepted")].copy()
    available_metrics = [column for column in METRICS if column in accepted]
    variant = ["actuation_scope", "scope_cutoff", "neighbor_k", "graph_candidate_k"]
    accepted = accepted[KEY + variant + available_metrics]
    budgets = budgets[KEY + variant + ["single_trajectory_success", "expected_denoiser_calls_per_valid"]]
    # Retry jobs can coexist with partial earlier runs.  They use the same
    # seed/noise and are not extra replications, so retain the newest source
    # before joining rather than double-counting them.
    accepted = accepted.drop_duplicates(KEY + variant, keep="last")
    budgets = budgets.drop_duplicates(KEY + variant, keep="last")
    runs = accepted.merge(
        budgets, on=KEY + variant, how="inner", validate="one_to_one"
    )
    # Identical all-scope runs appear only once per schedule, while older
    # aggregations can coexist on disk.  Keep one exact experimental row.
    runs = runs.drop_duplicates(KEY + variant)
    baseline = runs[runs["actuation_scope"] == "all"].copy()
    alternatives = runs[runs["actuation_scope"] != "all"].copy()
    paired = alternatives.merge(baseline, on=KEY, suffixes=("", "_all"), validate="many_to_one")
    for column in available_metrics + ["single_trajectory_success", "expected_denoiser_calls_per_valid"]:
        paired[f"delta_{column}"] = paired[column] - paired[f"{column}_all"]
    paired.to_csv(args.output / "paired_rows.csv", index=False)

    group = [
        "dataset", "catalog", "split", "profile", "peak", "floor", "cutoff",
        "actuation_scope", "scope_cutoff", "neighbor_k", "graph_candidate_k",
    ]
    rows = []
    rng = np.random.default_rng(20260929)
    delta_columns = [column for column in paired if column.startswith("delta_")]
    for values, block in paired.groupby(group, dropna=False):
        row = dict(zip(group, values))
        row["paired_cases"] = len(block)
        row["seeds"] = block["seed"].nunique()
        for column in delta_columns:
            vals = block[column].to_numpy(float)
            # Contracts generated from one trained adapter share a seed.  Use
            # seed-cluster means for uncertainty instead of treating every
            # condition as an independent replication.
            cluster_vals = block.groupby("seed")[column].mean().to_numpy(float)
            lo, hi = bootstrap(cluster_vals, rng)
            finite = vals[np.isfinite(vals)]
            row[column] = float(finite.mean()) if len(finite) else np.nan
            row[f"{column}_ci_low"] = lo
            row[f"{column}_ci_high"] = hi
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)

    ranked = summary.copy()
    ranked["fidelity_delta"] = ranked.get("delta_numeric_ks", 0) + ranked.get("delta_categorical_tv", 0)
    ranked = ranked.sort_values(["fidelity_delta", "delta_single_trajectory_success"], ascending=[True, False])
    lines = [
        "# Paired analysis of guidance scope",
        "",
        (f"Matched {len(paired)} condition/seed pairs against an all-feature guidance run. "
         "Negative KS/TV deltas mean better conditional fidelity; positive success delta means better condition satisfaction."),
        "",
        "| Dataset | Catalog | Split | Schedule | Scope | Switch t | k | Pool | Pairs | Δ success | Δ calls/valid | Δ KS | Δ TV |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked.head(80).itertuples(index=False):
        switch = "-" if pd.isna(row.scope_cutoff) else f"{row.scope_cutoff:g}"
        neighbor_k = "-" if pd.isna(row.neighbor_k) else f"{row.neighbor_k:g}"
        graph_pool = "-" if pd.isna(row.graph_candidate_k) else f"{row.graph_candidate_k:g}"
        lines.append(
            f"| {row.dataset} | {row.catalog} | {row.split} | {row.profile} p={row.peak:g} | "
            f"{row.actuation_scope} | {switch} | {neighbor_k} | {graph_pool} | {row.paired_cases} | "
            f"{row.delta_single_trajectory_success:+.4f} | {row.delta_expected_denoiser_calls_per_valid:+.2f} | "
            f"{getattr(row, 'delta_numeric_ks', float('nan')):+.4f} | "
            f"{getattr(row, 'delta_categorical_tv', float('nan')):+.4f} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
