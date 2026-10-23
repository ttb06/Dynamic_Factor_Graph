#!/usr/bin/env python3
"""Aggregate paired main runs, temporal ablations, and quality metrics."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


MAIN_RE = re.compile(r"main_state_(?P<dataset>.+)_s(?P<seed>\d+)_(?P<variant>dynamic_current|dynamic_stable|frozen_state)_")
WINDOW_RE = re.compile(r"stable_window_(?P<dataset>.+)_(?P<window>early|middle|late)_(?P<variant>dynamic|frozen_state)_")
QUALITY_RE = re.compile(r"main_quality_(?P<dataset>.+)_s(?P<seed>\d+)_")


def markdown(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    header = [str(x) for x in values.columns]
    rows = [[str(x) for x in row] for row in values.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(header) + " |",
                      "| " + " | ".join(["---"] * len(header)) + " |",
                      *("| " + " | ".join(row) + " |" for row in rows)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--main-job-id", required=True)
    parser.add_argument("--window-job-id", required=True)
    parser.add_argument("--quality-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    outputs = args.analysis / "outputs"

    main_rows = []; action_rows = []
    for path in outputs.glob(f"main_state_*_{args.main_job_id}_*/method_metrics.csv"):
        match = MAIN_RE.match(path.parent.name)
        if not match: continue
        frame = pd.read_csv(path); row = frame[frame.method.eq("x0_reward_calibrated")].iloc[0].to_dict()
        main_rows.append({**match.groupdict(), **row})
        if match.group("variant") == "dynamic_stable":
            for action_row in frame.to_dict("records"):
                if action_row["method"] == "manual_no_action": continue
                action_rows.append({**match.groupdict(), **action_row})
    main_frame = pd.DataFrame(main_rows); main_frame.to_csv(args.output / "main_per_run.csv", index=False)
    key_metrics = ["actuate_macro_violation", "actuate_heldout_violation_error", "heldout_violation_error",
                   "numeric_ks", "categorical_tv", "numeric_drift_sd", "action_events"]
    summary = main_frame.groupby(["dataset", "variant"])[key_metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in summary.columns]
    summary.to_csv(args.output / "main_mean_std.csv", index=False)
    action_frame = pd.DataFrame(action_rows); action_frame.to_csv(args.output / "action_ablation_per_run.csv", index=False)
    action_summary = action_frame.groupby(["dataset", "method"])[key_metrics].agg(["mean", "std"]).reset_index()
    action_summary.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in action_summary.columns]
    action_summary.to_csv(args.output / "action_ablation_mean_std.csv", index=False)

    window_rows = []
    for path in outputs.glob(f"stable_window_*_{args.window_job_id}_*/method_metrics.csv"):
        match = WINDOW_RE.match(path.parent.name)
        if not match: continue
        row = pd.read_csv(path).query("method == 'x0_reward_calibrated'").iloc[0].to_dict()
        window_rows.append({**match.groupdict(), **row})
    window_frame = pd.DataFrame(window_rows); window_frame.to_csv(args.output / "temporal_windows.csv", index=False)

    quality_rows = []
    for path in outputs.glob(f"main_quality_*_{args.quality_job_id}_*/baseline_metrics.csv"):
        match = QUALITY_RE.match(path.parent.name)
        if not match: continue
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"): quality_rows.append({**match.groupdict(), **row})
    quality_frame = pd.DataFrame(quality_rows); quality_frame.to_csv(args.output / "quality_per_run.csv", index=False)
    quality_metrics = ["shape_error_percent", "trend_error_percent", "alpha_precision_percent",
                       "beta_recall_percent", "detection_score", "mle_xgboost_auc",
                       "actuate_heldout_violation_error", "heldout_violation_error"]
    quality_summary = quality_frame.groupby(["dataset", "method"])[quality_metrics].agg(["mean", "std"]).reset_index()
    quality_summary.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in quality_summary.columns]
    quality_summary.to_csv(args.output / "quality_mean_std.csv", index=False)

    lines = ["# Submission-sprint aggregate", "", "## Main dynamic-state results", "",
             markdown(summary), "", "## Action-layer ablation", "", markdown(action_summary), "",
             "## Stable temporal-window ablation", "",
             markdown(window_frame[["dataset", "window", "variant", "actuate_macro_violation",
                                    "actuate_heldout_violation_error", "heldout_violation_error", "action_events"]]),
             "", "## General quality", "", markdown(quality_summary), ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
