#!/usr/bin/env python3
"""Compare budget-capped dynamic middle guidance against paired frozen-state runs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RUN_RE = re.compile(r"budget_match_(?P<dataset>.+)_middle_s(?P<seed>\d+)_(?P<job>\d+)_(?P<task>\d+)")
REF_TASK = {("shoppers", "20261031"): 0, ("shoppers", "20261032"): 1, ("shoppers", "20261033"): 2,
            ("default", "20261031"): 6, ("default", "20261032"): 7, ("default", "20261033"): 8}
METRICS = ["actuate_macro_violation", "actuate_heldout_violation_error", "heldout_violation_error",
           "numeric_ks", "categorical_tv", "numeric_drift_sd", "action_events", "runtime_seconds"]


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
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--reference-job-id", default="1060317")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    output_root = args.analysis / "outputs"
    rows = []
    for path in output_root.glob(f"budget_match_*_{args.job_id}_*/method_metrics.csv"):
        match = RUN_RE.fullmatch(path.parent.name)
        if match is None: continue
        dataset, seed = match.group("dataset"), match.group("seed")
        task = REF_TASK[(dataset, seed)]
        reference = output_root / f"time_seed_{dataset}_middle_s{seed}_frozen_state_{args.reference_job_id}_{task}" / "method_metrics.csv"
        dynamic, frozen = select(path), select(reference)
        record = {"dataset": dataset, "seed": seed}
        for metric in METRICS:
            record[f"dynamic_{metric}"] = dynamic[metric]
            record[f"frozen_{metric}"] = frozen[metric]
            record[f"improvement_{metric}"] = frozen[metric] - dynamic[metric]
        rows.append(record)
    per_seed = pd.DataFrame(rows)
    if len(per_seed) != 6: raise RuntimeError(f"Expected six paired runs, found {len(per_seed)}")
    per_seed.to_csv(args.output / "per_seed.csv", index=False)
    summary_rows = []
    for dataset, group in per_seed.groupby("dataset"):
        row = {"dataset": dataset}
        for metric in METRICS:
            row[f"dynamic_{metric}"] = group[f"dynamic_{metric}"].mean()
            row[f"frozen_{metric}"] = group[f"frozen_{metric}"].mean()
            row[f"improvement_{metric}"] = group[f"improvement_{metric}"].mean()
            row[f"win_rate_{metric}"] = (group[f"improvement_{metric}"] > 0).mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows); summary.to_csv(args.output / "summary.csv", index=False)
    columns = ["dataset", "dynamic_action_events", "frozen_action_events",
               "dynamic_actuate_heldout_violation_error", "frozen_actuate_heldout_violation_error",
               "improvement_actuate_heldout_violation_error", "win_rate_actuate_heldout_violation_error",
               "dynamic_heldout_violation_error", "frozen_heldout_violation_error",
               "improvement_heldout_violation_error", "win_rate_heldout_violation_error", "improvement_numeric_ks"]
    text = ["# Action-budget matched middle guidance", "",
            "Positive improvement is `frozen-state minus budget-capped dynamic`.", "", markdown(summary[columns]), ""]
    (args.output / "RESULTS.md").write_text("\n".join(text))


if __name__ == "__main__": main()
