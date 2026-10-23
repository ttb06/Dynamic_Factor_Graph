#!/usr/bin/env python3
"""Aggregate the matched dynamic-graph repaint screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.5f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--sample-job", required=True); parser.add_argument("--eval-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for directory in sorted(args.outputs.glob(f"dg_repaint_*_{args.sample_job}_*")):
        metadata_path = directory / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text()); task = directory.name.rsplit("_", 1)[-1]
        eval_matches = list(args.outputs.glob(f"dg_repaint_eval_*_{args.eval_job}_{task}/baseline_metrics.csv"))
        if not eval_matches:
            continue
        frame = pd.read_csv(eval_matches[0]); base = frame[frame.method == "tabdiff"].iloc[0]
        final = frame[frame.method == "final_repair"].iloc[0]
        for row in frame.itertuples(index=False):
            record = row._asdict(); record.update({
                "run": directory.name, "dataset": metadata["dataset"], "trigger": metadata["trigger"],
                "checkpoint": metadata["actual_checkpoint"], "eligible_edges": metadata["eligible_edges"],
                "heldout_gain_vs_base_pct": 100 * (base.heldout_violation_error - row.heldout_violation_error)
                / max(base.heldout_violation_error, 1e-12),
                "heldout_gain_vs_final_pct": 100 * (final.heldout_violation_error - row.heldout_violation_error)
                / max(final.heldout_violation_error, 1e-12),
                "shape_delta_vs_final": row.shape_error_percent - final.shape_error_percent,
                "trend_delta_vs_final": row.trend_error_percent - final.trend_error_percent,
                "auc_delta_vs_final": row.mle_xgboost_auc - final.mle_xgboost_auc,
            })
            records.append(record)
    if not records:
        raise RuntimeError("no completed matched evaluation found")
    result = pd.DataFrame(records); result.to_csv(args.output / "all_metrics.csv", index=False)
    focus = result[result.method.str.contains("repaint")].copy()
    columns = ["dataset", "trigger", "checkpoint", "method", "heldout_violation_error",
               "heldout_gain_vs_final_pct", "shape_error_percent", "shape_delta_vs_final",
               "trend_error_percent", "trend_delta_vs_final", "mle_xgboost_auc", "auc_delta_vs_final"]
    lines = ["# Dynamic graph conditional repaint screen", "", table(focus[columns]), "",
             "Positive gain versus final means lower held-out graph error than calibrated final repair. ",
             "Shape/Trend deltas must be non-positive and AUC delta should be non-negative for strict dominance."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
