#!/usr/bin/env python3
"""Aggregate row-assignment and refresh-frequency controls against paired references."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SEEDS = ["20261031", "20261032", "20261033"]
TASKS = {"shoppers": [0, 1, 2], "default": [6, 7, 8]}
METRICS = ["actuate_heldout_violation_error", "heldout_violation_error", "numeric_ks",
           "categorical_tv", "numeric_drift_sd", "action_events", "runtime_seconds"]


def select(path: Path) -> dict:
    frame = pd.read_csv(path)
    return frame.loc[frame.method.eq("x0_reward_calibrated")].iloc[0].to_dict()


def markdown(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda value: f"{value:.6f}")
    rows = [[str(value) for value in row] for row in values.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(map(str, values.columns)) + " |",
                      "| " + " | ".join(["---"] * len(values.columns)) + " |",
                      *("| " + " | ".join(row) + " |" for row in rows)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--shuffled-job-id", required=True)
    parser.add_argument("--periodic-job-id", required=True)
    parser.add_argument("--reference-job-id", default="1060317")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    root = args.analysis / "outputs"; rows = []
    for dataset in ["shoppers", "default"]:
        for ordinal, seed in enumerate(SEEDS):
            reference_task = TASKS[dataset][ordinal]
            for variant in ["dynamic", "frozen_state"]:
                path = root / f"time_seed_{dataset}_middle_s{seed}_{variant}_{args.reference_job_id}_{reference_task}" / "method_metrics.csv"
                rows.append({**select(path), "dataset": dataset, "seed": seed, "method": variant})
            shuffled = next(root.glob(f"shuffled_row_{dataset}_middle_s{seed}_{args.shuffled_job_id}_*/method_metrics.csv"))
            rows.append({**select(shuffled), "dataset": dataset, "seed": seed, "method": "shuffled_row"})
            for interval in [3, 5, 10]:
                periodic = next(root.glob(f"periodic_k{interval}_{dataset}_middle_s{seed}_{args.periodic_job_id}_*/method_metrics.csv"))
                rows.append({**select(periodic), "dataset": dataset, "seed": seed, "method": f"periodic_k{interval}"})
    per_seed = pd.DataFrame(rows); per_seed.to_csv(args.output / "per_seed.csv", index=False)
    summary = per_seed.groupby(["dataset", "method"])[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(value).rstrip("_") if isinstance(value, tuple) else value for value in summary.columns]
    summary.to_csv(args.output / "mean_std.csv", index=False)
    columns = ["dataset", "method", "actuate_heldout_violation_error_mean",
               "actuate_heldout_violation_error_std", "heldout_violation_error_mean",
               "numeric_ks_mean", "categorical_tv_mean", "action_events_mean", "runtime_seconds_mean"]
    lines = ["# Dynamic-topology mechanism controls", "",
             "All methods use the same three seeds and middle intervention window.", "",
             markdown(summary[columns]), ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__": main()
