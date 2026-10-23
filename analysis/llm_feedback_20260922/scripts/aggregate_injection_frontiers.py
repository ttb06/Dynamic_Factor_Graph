#!/usr/bin/env python3
"""Aggregate coarse injection sweeps without selecting on the audit metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def markdown(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown renderer for cluster environments."""
    def cell(value):
        if isinstance(value, float):
            return f"{value:.5f}"
        return str(value)
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--job-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    prefixes = ("wide_strength_", "action_intensity_", "window_strength_", "graph_reward_",
                "feedback_frontier_", "feedback_window_", "feedback_topology_",
                "promotion_frontier_", "promotion_topology_", "feedback_uncapped_",
                "promotion_uncapped_", "promotion_uncapped_topology_",
                "interior_margin_", "interior_margin_topology_",
                "promotion_calibration_", "margin_promotion_")
    for directory in sorted(args.outputs.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(prefixes):
            continue
        if not any(job_id in directory.name for job_id in args.job_ids):
            continue
        metric_path = directory / "method_metrics.csv"
        metadata_path = directory / "run_metadata.json"
        if not metric_path.exists() or not metadata_path.exists():
            continue
        metrics = pd.read_csv(metric_path)
        guided = metrics.loc[metrics.method == "x0_reward_calibrated"].iloc[0]
        base = metrics.loc[metrics.method == "unconditional"].iloc[0]
        metadata = json.loads(metadata_path.read_text())
        row = {
            "run": directory.name,
            "dataset": metadata["dataset"],
            "macro_error": guided.macro_violation,
            "heldout_error": guided.heldout_violation_error,
            "calibration_error": guided.calibration_violation_error,
            "numeric_ks": guided.numeric_ks,
            "categorical_tv": guided.categorical_tv,
            "action_events": int(guided.action_events),
            "rows_actuated": int(guided.unique_rows_actuated),
            "macro_gain_pct": 100 * (base.macro_violation - guided.macro_violation) / base.macro_violation,
            "heldout_gain_pct": 100 * (base.heldout_violation_error - guided.heldout_violation_error) / base.heldout_violation_error,
            "calibration_gain_pct": 100 * (base.calibration_violation_error - guided.calibration_violation_error) / base.calibration_violation_error,
            "numeric_guidance_scale": metadata["numeric_guidance_scale"],
            "graph_weight": metadata["graph_weight"],
            "max_actions_per_row_edge": metadata["max_actions_per_row_edge"],
            "cooldown_steps": metadata["cooldown_steps"],
            "action_hold_steps": metadata["action_hold_steps"],
            "active_min_t": metadata["active_min_t"],
            "active_max_t": metadata["active_max_t"],
            "numeric_guidance_mode": metadata.get("numeric_guidance_mode"),
            "feedback_repeat_growth": metadata.get("feedback_repeat_growth"),
            "feedback_late_multiplier": metadata.get("feedback_late_multiplier"),
            "max_guidance_delta": metadata.get("max_guidance_delta"),
            "action_margin_sd": metadata.get("action_margin_sd"),
            "promote_score_only": metadata.get("promote_score_only"),
            "promoted_min_streak": metadata.get("promoted_min_streak"),
            "promoted_max_calibration_rate": metadata.get("promoted_max_calibration_rate"),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "all_frontiers.csv", index=False)
    if frame.empty:
        (args.output / "RESULTS.md").write_text("# Injection frontiers\n\nNo completed runs found.\n")
        return

    # Rank only for reading convenience.  Confirmation must use a new seed;
    # this table is explicitly exploratory and is not a final model selection.
    ranked = frame.sort_values(["dataset", "macro_error", "heldout_error"])
    ranked.to_csv(args.output / "ranked_exploratory.csv", index=False)
    columns = ["dataset", "run", "macro_error", "macro_gain_pct", "heldout_error",
               "heldout_gain_pct", "calibration_error", "calibration_gain_pct", "numeric_ks",
               "categorical_tv", "action_events"]
    lines = [
        "# Coarse injection frontiers",
        "",
        "Exploratory one-seed sweep. Gains are relative error reductions against the paired unconditional sample. Do not report the best row as a confirmed result.",
        "",
        markdown(ranked[columns]),
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
