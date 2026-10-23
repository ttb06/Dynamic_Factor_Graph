#!/usr/bin/env python3
"""Aggregate paired dynamic/frozen temporal-window runs across sampling seeds."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RUN_RE = re.compile(
    r"time_seed_(?P<dataset>.+)_(?P<window>middle|late)_s(?P<seed>\d+)_"
    r"(?P<variant>dynamic|frozen_state)_(?P<job>\d+)_(?P<task>\d+)"
)
METRICS = [
    "actuate_macro_violation",
    "actuate_heldout_violation_error",
    "heldout_violation_error",
    "numeric_ks",
    "categorical_tv",
    "numeric_drift_sd",
    "action_events",
    "runtime_seconds",
]


def markdown(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    header = [str(value) for value in values.columns]
    rows = [[str(value) for value in row] for row in values.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    for path in (args.analysis / "outputs").glob(f"time_seed_*_{args.job_id}_*/method_metrics.csv"):
        match = RUN_RE.fullmatch(path.parent.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        selected = frame.loc[frame["method"].eq("x0_reward_calibrated")]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one x0_reward_calibrated row in {path}")
        rows.append({**match.groupdict(), **selected.iloc[0].to_dict()})
    per_run = pd.DataFrame(rows)
    expected = 24  # 2 datasets x 2 windows x 3 seeds x 2 variants
    if len(per_run) != expected:
        raise RuntimeError(f"Expected {expected} completed outputs, found {len(per_run)}")
    per_run.to_csv(args.output / "per_run.csv", index=False)

    summary = per_run.groupby(["dataset", "window", "variant"])[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(value).rstrip("_") if isinstance(value, tuple) else value for value in summary.columns]
    summary.to_csv(args.output / "mean_std.csv", index=False)

    index = ["dataset", "window", "seed"]
    wide = per_run.pivot(index=index, columns="variant", values=METRICS)
    delta_rows = []
    for key, row in wide.iterrows():
        record = dict(zip(index, key))
        for metric in METRICS:
            # Positive means dynamic is better for error/cost metrics.
            record[f"improvement_{metric}"] = row[(metric, "frozen_state")] - row[(metric, "dynamic")]
        delta_rows.append(record)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(args.output / "paired_deltas.csv", index=False)
    delta_metrics = [column for column in deltas if column.startswith("improvement_")]
    paired = deltas.groupby(["dataset", "window"])[delta_metrics].agg(["mean", "std", lambda values: (values > 0).mean()]).reset_index()
    paired.columns = [
        "_".join(str(part) for part in value if part).replace("<lambda_0>", "win_rate").rstrip("_")
        if isinstance(value, tuple) else value
        for value in paired.columns
    ]
    paired.to_csv(args.output / "paired_summary.csv", index=False)

    report_columns = [
        "dataset", "window",
        "improvement_actuate_heldout_violation_error_mean",
        "improvement_actuate_heldout_violation_error_std",
        "improvement_actuate_heldout_violation_error_win_rate",
        "improvement_heldout_violation_error_mean",
        "improvement_heldout_violation_error_win_rate",
        "improvement_numeric_ks_mean",
        "improvement_action_events_mean",
    ]
    lines = [
        "# Multi-seed temporal-window comparison",
        "",
        "Each improvement is `frozen-state minus dynamic`; positive values favor the dynamic graph.",
        "",
        markdown(paired[report_columns]),
        "",
        "## Mean and standard deviation by method",
        "",
        markdown(summary),
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
