#!/usr/bin/env python3
"""Aggregate parity-valid row-wise mechanism ablations and cutoff sensitivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", required=True,
                        help="Comma-separated label:job pairs, e.g. full:1,once:2")
    parser.add_argument("--trajectory-jobs", required=True,
                        help="Comma-separated label:job pairs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    for item in args.jobs.split(","):
        label, job = item.split(":", 1)
        directory = args.root / f"rowwise_reverse_action_{job}"
        metadata = json.loads((directory / "run_metadata.json").read_text())
        frame = pd.read_csv(directory / "method_metrics.csv")
        frame.insert(0, "job_id", job); frame.insert(0, "ablation", label)
        frame["max_actions_per_row_edge"] = metadata.get("max_actions_per_row_edge", 0)
        frame["cooldown_steps"] = metadata.get("cooldown_steps", 0)
        rows.append(frame)
    metrics = pd.concat(rows, ignore_index=True)
    metrics.to_csv(args.output / "mechanism_ablation.csv", index=False)

    schedules = []
    for item in args.trajectory_jobs.split(","):
        label, job = item.split(":", 1)
        frame = pd.read_csv(args.root / f"rowwise_mechanism_trajectory_{job}" / "typed_readability_schedule.csv")
        frame.insert(0, "job_id", job); frame.insert(0, "probe", label); schedules.append(frame)
    schedule = pd.concat(schedules, ignore_index=True)
    schedule.to_csv(args.output / "cutoff_sensitivity.csv", index=False)
    pivot = schedule.pivot(index="candidate_id", columns="probe", values="cutoff")
    if len(pivot.columns) >= 2:
        pivot["range"] = pivot.max(axis=1) - pivot.min(axis=1)
    pivot.reset_index().to_csv(args.output / "cutoff_ranges.csv", index=False)

    focus = metrics[metrics.method.isin([
        "unconditional", "manual_no_action", "xstate_directed_calibrated",
        "x0_directed_calibrated", "x0_reward_calibrated",
        "rollback_reward_calibrated",
        "final_directed_calibrated", "final_reward_calibrated",
    ])].copy()
    lines = [
        "# Row-wise mechanism ablation", "",
        "All runs below passed exact no-action parity. Runs with 4,096 samples share the same base sample and differ only in feedback budget or cutoff schedule.", "",
        "| Ablation | Method | Violation | Heldout error | Calibration error | KS | Drift | Affected drift | Density delta | Actions | Rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.ablation} | {row.method} | {row.macro_violation:.4f} | "
            f"{row.heldout_violation_error:.4f} | {row.calibration_violation_error:.4f} | "
            f"{row.numeric_ks:.4f} | {row.numeric_drift_sd:.4f} | "
            f"{row.affected_numeric_drift_sd:.4f} | {row.affected_density_delta:.4f} | "
            f"{int(row.action_events)} | {int(row.unique_rows_actuated)} |"
        )
    lines += ["", "## Cutoff sensitivity", "",
              "| Edge | " + " | ".join(pivot.columns.astype(str)) + " |",
              "| --- | " + " | ".join(["---:"] * len(pivot.columns)) + " |"]
    for edge, row in pivot.iterrows():
        vals = [f"{value:.3f}" if np.isfinite(value) else "NA" for value in row]
        lines.append(f"| {edge} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "Interpretation: lower violation alone is not sufficient. A middle-step method must improve heldout/calibration error or affected-row density relative to the matched final repair at comparable action count.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
