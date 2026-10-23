#!/usr/bin/env python3
"""Aggregate dynamic-constraint sweeps and expose the decisive error slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def safe_rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def metrics(frame: pd.DataFrame) -> dict:
    valid = ~frame["missing_decision"].astype(bool)
    gold_v = frame["gold_status"].eq("violated")
    pred_v = frame["pred_status"].eq("violated")
    tp = int((gold_v & pred_v).sum())
    fp = int((~gold_v & pred_v).sum())
    fn = int((gold_v & ~pred_v).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "cases": len(frame),
        "applicability_accuracy": safe_rate(
            frame.loc[valid, "gold_applicability"].eq(frame.loc[valid, "pred_applicability"])
        ),
        "status_accuracy": safe_rate(frame.loc[valid, "gold_status"].eq(frame.loc[valid, "pred_status"])),
        "violation_precision": precision,
        "violation_recall": recall,
        "violation_f1": 2 * precision * recall / (precision + recall) if precision + recall else float("nan"),
        "nonapplicable_false_action_rate": safe_rate(
            frame.loc[frame.gold_status.eq("not_applicable"), "action"].eq("rollback_candidate")
        ),
        "satisfied_false_action_rate": safe_rate(
            frame.loc[frame.gold_status.eq("satisfied"), "action"].eq("rollback_candidate")
        ),
        "missing_rate": safe_rate(frame["missing_decision"].astype(bool)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    requested = [item.strip() for item in args.job_ids.split(",") if item.strip()]
    frames, summaries, missing = [], {}, []
    for job_id in requested:
        matches = sorted(args.root.glob(f"dynamic_constraint_*_{job_id}"))
        if not matches or not (matches[0] / "decisions.csv").exists():
            missing.append(job_id)
            continue
        directory = matches[0]
        experiment = directory.name.removesuffix(f"_{job_id}").removeprefix("dynamic_constraint_")
        frame = pd.read_csv(directory / "decisions.csv", keep_default_na=False)
        frame["experiment"] = experiment
        frame["source_job"] = job_id
        frames.append(frame)
        summary_path = directory / "summary.json"
        summaries[experiment] = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if not frames:
        raise RuntimeError("No completed dynamic-constraint outputs found")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output / "combined_decisions.csv", index=False)
    overall_rows, slice_rows = [], []
    for experiment, frame in combined.groupby("experiment"):
        row = {"experiment": experiment, **metrics(frame)}
        row.update({key: summaries[experiment].get(key, 0) for key in ("input_tokens", "output_tokens", "total_tokens", "latency_seconds")})
        overall_rows.append(row)
        for (constraint_id, state, t_value), subset in frame.groupby(["constraint_id", "state", "t"]):
            slice_rows.append({
                "experiment": experiment,
                "constraint_id": constraint_id,
                "state": state,
                "t": t_value,
                **metrics(subset),
            })
    overall = pd.DataFrame(overall_rows).sort_values(["violation_f1", "applicability_accuracy"], ascending=False)
    slices = pd.DataFrame(slice_rows)
    overall.to_csv(args.output / "overall_metrics.csv", index=False)
    slices.to_csv(args.output / "metrics_by_constraint_state_timestep.csv", index=False)
    slices.groupby(["experiment", "constraint_id"], as_index=False).agg(
        cases=("cases", "sum"),
        applicability_accuracy=("applicability_accuracy", "mean"),
        status_accuracy=("status_accuracy", "mean"),
        violation_precision=("violation_precision", "mean"),
        violation_recall=("violation_recall", "mean"),
        nonapplicable_false_action_rate=("nonapplicable_false_action_rate", "mean"),
        satisfied_false_action_rate=("satisfied_false_action_rate", "mean"),
    ).to_csv(args.output / "metrics_by_constraint.csv", index=False)
    metadata = {
        "status": "executed",
        "jobs_requested": requested,
        "jobs_loaded": sorted(combined.source_job.astype(str).unique().tolist()),
        "missing_jobs": missing,
        "interpretation": "This benchmark separates applicability, satisfaction, and action safety on synthetic previews.",
        "privacy_boundary": "No real rows were sent to an API.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
