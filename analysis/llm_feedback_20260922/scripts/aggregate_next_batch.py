#!/usr/bin/env python3
"""Aggregate Diabetes mechanism runs and action-matched row-assignment controls."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["actuate_heldout_violation_error", "heldout_violation_error", "numeric_ks",
           "categorical_tv", "numeric_drift_sd", "action_events", "runtime_seconds"]


def select(path: Path) -> dict:
    frame = pd.read_csv(path)
    return frame.loc[frame.method.eq("x0_reward_calibrated")].iloc[0].to_dict()


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = frame.groupby(keys)[METRICS].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(value).rstrip("_") if isinstance(value, tuple) else value for value in result.columns]
    return result


def markdown(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    rows = [[str(value) for value in row] for row in values.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(map(str, values.columns)) + " |",
                      "| " + " | ".join(["---"] * len(values.columns)) + " |",
                      *("| " + " | ".join(row) + " |" for row in rows)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--diabetes-full-job", required=True)
    parser.add_argument("--diabetes-middle-job", required=True)
    parser.add_argument("--row-budget-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    root = args.analysis / "outputs"

    diabetes_full = []
    for seed in ["20261026", "20261027", "20261028"]:
        for variant in ["dynamic_current", "dynamic_stable", "frozen_state"]:
            path = next(root.glob(f"diabetes_state_s{seed}_{variant}_{args.diabetes_full_job}_*/method_metrics.csv"))
            diabetes_full.append({**select(path), "seed": seed, "variant": variant})
    diabetes_full = pd.DataFrame(diabetes_full); diabetes_full.to_csv(args.output / "diabetes_full_per_seed.csv", index=False)
    diabetes_full_summary = summarize(diabetes_full, ["variant"])
    diabetes_full_summary.to_csv(args.output / "diabetes_full_summary.csv", index=False)

    diabetes_middle = []
    for seed in ["20261026", "20261027", "20261028"]:
        for variant in ["dynamic", "frozen_state"]:
            path = next(root.glob(f"diabetes_middle_s{seed}_{variant}_{args.diabetes_middle_job}_*/method_metrics.csv"))
            diabetes_middle.append({**select(path), "seed": seed, "variant": variant})
    diabetes_middle = pd.DataFrame(diabetes_middle); diabetes_middle.to_csv(args.output / "diabetes_middle_per_seed.csv", index=False)
    diabetes_middle_summary = summarize(diabetes_middle, ["variant"])
    diabetes_middle_summary.to_csv(args.output / "diabetes_middle_summary.csv", index=False)

    row_budget = []
    for dataset in ["shoppers", "default"]:
        for seed in ["20261031", "20261032", "20261033"]:
            for variant in ["dynamic", "shuffled"]:
                path = next(root.glob(f"row_budget40_{dataset}_s{seed}_{variant}_{args.row_budget_job}_*/method_metrics.csv"))
                row_budget.append({**select(path), "dataset": dataset, "seed": seed, "variant": variant})
    row_budget = pd.DataFrame(row_budget); row_budget.to_csv(args.output / "row_budget_per_seed.csv", index=False)
    row_budget_summary = summarize(row_budget, ["dataset", "variant"])
    row_budget_summary.to_csv(args.output / "row_budget_summary.csv", index=False)

    columns = ["actuate_heldout_violation_error_mean", "actuate_heldout_violation_error_std",
               "heldout_violation_error_mean", "numeric_ks_mean", "categorical_tv_mean",
               "action_events_mean", "runtime_seconds_mean"]
    lines = ["# Next-batch aggregate", "", "## Diabetes full window", "",
             markdown(diabetes_full_summary[["variant", *columns]]), "", "## Diabetes middle window", "",
             markdown(diabetes_middle_summary[["variant", *columns]]), "", "## Row assignment at 40-action budget", "",
             markdown(row_budget_summary[["dataset", "variant", *columns]]), ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__": main()
