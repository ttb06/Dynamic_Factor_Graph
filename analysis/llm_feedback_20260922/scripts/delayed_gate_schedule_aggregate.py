#!/usr/bin/env python3
"""Aggregate delayed applicability-gate schedule probes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def markdown(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value): cells.append("")
            elif isinstance(value, float): cells.append(f"{value:.4f}")
            else: cells.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    wanted = {x for x in re.split(r"[:,]", args.job_ids) if x}
    frames, meta = [], []
    for path in args.run_root.glob("delayed_gate_schedule_*"):
        if not any(path.name.endswith(f"_{job}") for job in wanted): continue
        frames.append(pd.read_csv(path / "schedule_metrics.csv"))
        meta.append(json.loads((path / "run_metadata.json").read_text()))
    if not frames: raise RuntimeError("no schedule results found")
    all_rows = pd.concat(frames, ignore_index=True); all_rows.to_csv(args.output / "all_schedule_metrics.csv", index=False)
    by_seed = all_rows.groupby(["seed", "schedule", "gate_start_t", "split"]).agg(
        conditional_violation=("conditional_violation_rate", "mean"), violation_reduction=("conditional_violation_reduction", "mean"),
        antecedent_error=("antecedent_rate_error", "mean"), exception_error=("exception_rate_error", "mean"),
        inactive_target_error=("inactive_target_mean_error", "mean"), numeric_ks=("global_numeric_ks", "mean"),
        categorical_tv=("global_categorical_tv", "mean"),
    ).reset_index(); by_seed.to_csv(args.output / "summary_by_seed.csv", index=False)
    stability = by_seed.groupby(["schedule", "gate_start_t", "split"]).agg(
        seeds=("seed", "nunique"), violation=("conditional_violation", "mean"), violation_std=("conditional_violation", "std"),
        reduction=("violation_reduction", "mean"), antecedent_error=("antecedent_error", "mean"),
        exception_error=("exception_error", "mean"), inactive_error=("inactive_target_error", "mean"),
        numeric_ks=("numeric_ks", "mean"), categorical_tv=("categorical_tv", "mean"),
    ).reset_index().sort_values(["split", "gate_start_t"], ascending=[True, False])
    stability.to_csv(args.output / "stability.csv", index=False)
    text = ["# Delayed applicability-gate schedule", "", markdown(stability), "",
            "All schedules use the same trained adapter and paired initial noise. `never` is the no-actuation control; `always` activates the row gate from t=1."]
    (args.output / "RESULTS.md").write_text("\n".join(text) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({"job_ids": sorted(wanted), "runs": meta}, indent=2))
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    main()
