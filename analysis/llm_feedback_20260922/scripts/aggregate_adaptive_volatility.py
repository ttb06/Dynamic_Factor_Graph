#!/usr/bin/env python3
"""Aggregate volatility-adaptive graph refresh against fixed refresh policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "actuate_heldout_violation_error",
    "heldout_violation_error",
    "numeric_ks",
    "categorical_tv",
    "numeric_drift_sd",
    "action_events",
    "runtime_seconds",
]


def selected(directory: Path) -> dict:
    frame = pd.read_csv(directory / "method_metrics.csv")
    result = frame.loc[frame.method.eq("x0_reward_calibrated")].iloc[0].to_dict()
    trace = json.loads((directory / "trace_x0_reward_calibrated.json").read_text())
    metadata = json.loads((directory / "run_metadata.json").read_text())
    denominator = int(metadata["samples"]) * int(trace.get("topology_refreshes", 0))
    result["normalized_topology_flip_rate"] = (
        int(trace.get("topology_flips", 0)) / denominator if denominator else float("nan")
    )
    counts = trace.get("adaptive_persistence_counts", {})
    total = sum(int(value) for value in counts.values())
    result["adaptive_p3_share"] = int(counts.get("3", 0)) / total if total else float("nan")
    return result


def locate(root: Path, dataset: str, method: str, seed: str, args) -> Path:
    if method == "adaptive":
        return next(root.glob(f"adaptive_vol_{dataset}_adaptive_hysteresis_s{seed}_{args.adaptive_job}_*"))
    if dataset == "diabetes":
        if method == "dynamic":
            return next(root.glob(f"adaptive_vol_diabetes_dynamic_s{seed}_{args.adaptive_job}_*"))
        return next(root.glob(f"diabetes_hysteresis_p3_s{seed}_{args.diabetes_hysteresis_job}_*"))
    if method == "dynamic":
        index = {"shoppers": {"20261031": 0, "20261032": 1, "20261033": 2},
                 "default": {"20261031": 6, "20261032": 7, "20261033": 8}}[dataset][seed]
        return root / f"time_seed_{dataset}_middle_s{seed}_dynamic_{args.temporal_job}_{index}"
    index = {"shoppers": {"20261031": 0, "20261032": 1, "20261033": 2},
             "default": {"20261031": 3, "20261032": 4, "20261033": 5}}[dataset][seed]
    return root / f"hysteresis_p3_{dataset}_s{seed}_{args.sd_hysteresis_job}_{index}"


def markdown(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(map(str, values.columns)) + " |",
        "| " + " | ".join(["---"] * len(values.columns)) + " |",
        *rows,
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--adaptive-job", required=True)
    parser.add_argument("--temporal-job", default="1060317")
    parser.add_argument("--sd-hysteresis-job", default="1060464")
    parser.add_argument("--diabetes-hysteresis-job", default="1060478")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.analysis / "outputs"
    rows = []
    for dataset, seeds in {
        "shoppers": ["20261031", "20261032", "20261033"],
        "default": ["20261031", "20261032", "20261033"],
        "diabetes": ["20261041", "20261042", "20261043"],
    }.items():
        for seed in seeds:
            for method in ["dynamic", "hysteresis_p3", "adaptive"]:
                directory = locate(root, dataset, method, seed, args)
                rows.append({**selected(directory), "dataset": dataset, "seed": seed, "method_variant": method})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(args.output / "adaptive_per_seed.csv", index=False)
    summary = per_seed.groupby(["dataset", "method_variant"])[
        METRICS + ["adaptive_p3_share", "normalized_topology_flip_rate"]
    ].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(value).rstrip("_") if isinstance(value, tuple) else value for value in summary.columns]
    summary.to_csv(args.output / "adaptive_summary.csv", index=False)

    columns = [
        "dataset", "method_variant", "actuate_heldout_violation_error_mean",
        "actuate_heldout_violation_error_std", "heldout_violation_error_mean",
        "numeric_ks_mean", "categorical_tv_mean", "action_events_mean",
        "adaptive_p3_share_mean", "normalized_topology_flip_rate_mean", "runtime_seconds_mean",
    ]
    lines = [
        "# Volatility-adaptive graph refresh",
        "",
        "The adaptive policy uses persistence 3 when the online EMA of row-edge topology flips is at least 0.0015; otherwise it refreshes immediately.",
        "",
        markdown(summary[columns]),
        "",
    ]
    for dataset in ["shoppers", "default", "diabetes"]:
        view = per_seed.loc[per_seed.dataset.eq(dataset)].pivot(index="seed", columns="method_variant", values="actuate_heldout_violation_error")
        lines.append(
            f"- {dataset}: adaptive beats dynamic in {int((view.adaptive < view.dynamic).sum())}/3 seeds and fixed hysteresis-p3 in {int((view.adaptive < view.hysteresis_p3).sum())}/3 seeds."
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
