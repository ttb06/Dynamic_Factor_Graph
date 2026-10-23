#!/usr/bin/env python3
"""Aggregate focused Diabetes topology, timing, and hysteresis controls."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "actuate_heldout_violation_error",
    "heldout_violation_error",
    "numeric_ks",
    "categorical_tv",
    "numeric_drift_sd",
    "action_events",
    "unique_rows_actuated",
    "runtime_seconds",
]


def selected(path: Path) -> dict:
    frame = pd.read_csv(path)
    return frame.loc[frame.method.eq("x0_reward_calibrated")].iloc[0].to_dict()


def load(root: Path, pattern: str, seeds: list[str], variants: list[str]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for variant in variants:
            path = next(root.glob(pattern.format(seed=seed, variant=variant)))
            rows.append({**selected(path), "seed": seed, "variant": variant})
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.groupby("variant")[METRICS].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(value).rstrip("_") if isinstance(value, tuple) else value for value in result.columns]
    return result


def markdown(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    rows = [[str(value) for value in row] for row in values.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(map(str, values.columns)) + " |",
        "| " + " | ".join(["---"] * len(values.columns)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def paired_wins(frame: pd.DataFrame, reference: str, challenger: str, metric: str) -> str:
    pivot = frame.pivot(index="seed", columns="variant", values=metric)
    wins = int((pivot[challenger] < pivot[reference]).sum())
    return f"{challenger} tốt hơn {reference} ở {wins}/{len(pivot)} seed theo {metric}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--hysteresis-job", required=True)
    parser.add_argument("--topology-job", required=True)
    parser.add_argument("--timing-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.analysis / "outputs"
    seeds = ["20261041", "20261042", "20261043"]

    hysteresis = load(
        root,
        f"diabetes_hysteresis_{{variant}}_s{{seed}}_{args.hysteresis_job}_*/method_metrics.csv",
        seeds,
        ["p2", "p3"],
    )
    topology = load(
        root,
        f"diabetes_budget20_{{variant}}_s{{seed}}_{args.topology_job}_*/method_metrics.csv",
        seeds,
        ["dynamic", "frozen_state", "shuffled"],
    )
    timing = load(
        root,
        f"diabetes_timing_{{variant}}_budget20_s{{seed}}_{args.timing_job}_*/method_metrics.csv",
        seeds,
        ["early", "middle", "late"],
    )
    for name, frame in [("hysteresis", hysteresis), ("topology_budget20", topology), ("timing_budget20", timing)]:
        frame.to_csv(args.output / f"{name}_per_seed.csv", index=False)
        summarize(frame).to_csv(args.output / f"{name}_summary.csv", index=False)

    columns = [
        "actuate_heldout_violation_error_mean",
        "actuate_heldout_violation_error_std",
        "heldout_violation_error_mean",
        "numeric_ks_mean",
        "categorical_tv_mean",
        "action_events_mean",
        "unique_rows_actuated_mean",
        "runtime_seconds_mean",
    ]
    lines = [
        "# Diabetes focused controls",
        "",
        "## Hysteresis",
        "",
        markdown(summarize(hysteresis)[["variant", *columns]]),
        "",
        "## Topology under a 20-action cap",
        "",
        markdown(summarize(topology)[["variant", *columns]]),
        "",
        paired_wins(topology, "shuffled", "dynamic", "actuate_heldout_violation_error"),
        paired_wins(topology, "frozen_state", "dynamic", "actuate_heldout_violation_error"),
        "",
        "## Timing under a 20-action cap",
        "",
        markdown(summarize(timing)[["variant", *columns]]),
        "",
        paired_wins(timing, "early", "middle", "actuate_heldout_violation_error"),
        paired_wins(timing, "late", "middle", "actuate_heldout_violation_error"),
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
