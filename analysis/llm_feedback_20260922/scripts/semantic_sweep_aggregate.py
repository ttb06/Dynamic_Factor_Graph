#!/usr/bin/env python3
"""Aggregate semantic LLM jobs and search cost-aware timestep schedules."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import pandas as pd


def sets(value) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return set(filter(None, str(value).split("|")))


def score(predictions: list[set[str]], golds: list[set[str]]) -> dict[str, float | int]:
    tp = sum(len(pred & gold) for pred, gold in zip(predictions, golds))
    fp = sum(len(pred - gold) for pred, gold in zip(predictions, golds))
    fn = sum(len(gold - pred) for pred, gold in zip(predictions, golds))
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def experiment_metrics(frame: pd.DataFrame, metadata: dict) -> dict:
    preds = [sets(value) for value in frame["predicted"]]
    golds = [sets(value) for value in frame["gold"]]
    issue = [bool(gold) for gold in golds]
    flagged = [bool(pred) for pred in preds]
    rare = frame["control_type"].isin(["rare_valid_country", "exception_valid_own_child"])
    result = score(preds, golds)
    result.update(
        {
            "records": len(frame),
            "issue_detection_accuracy": sum(a == b for a, b in zip(issue, flagged)) / len(frame),
            "abstain_rate": float(frame["status"].eq("abstain").mean()),
            "missing_rate": float(frame["missing_decision"].astype(bool).mean()),
            "rare_control_false_positive_rate": float(frame.loc[rare, "predicted"].fillna("").ne("").mean()) if rare.any() else math.nan,
            "unnecessary_action_rate": sum(bool(pred) and not bool(gold) for pred, gold in zip(preds, golds)) / len(frame),
            "residual_issue_rate_after_perfect_rollback": result["fn"] / max(1, sum(map(len, golds))),
            "input_tokens": metadata.get("input_tokens", 0),
            "output_tokens": metadata.get("output_tokens", 0),
            "total_tokens": metadata.get("total_tokens", 0),
            "successful_calls": metadata.get("successful_calls", 0),
            "failed_calls": metadata.get("failed_calls", 0),
            "api_latency_seconds": metadata.get("api_latency_seconds", 0),
        }
    )
    result["tp_per_1k_tokens"] = 1000 * result["tp"] / result["total_tokens"] if result["total_tokens"] else math.nan
    return result


def schedule_rows(frame: pd.DataFrame, experiment: str) -> list[dict]:
    frame = frame[frame["repeat"].eq(0)].copy()
    times = sorted(frame["t"].unique(), reverse=True)
    rows = []
    for length in (1, 2, 3):
        for schedule in itertools.combinations(times, length):
            predictions, golds, calls = [], [], []
            grouped = frame[frame["t"].isin(schedule)].groupby(["semantic_key", "control_type"])
            for _, group in grouped:
                by_t = {float(row.t): row for row in group.itertuples()}
                if not all(float(t) in by_t for t in schedule):
                    continue
                chosen = set()
                used = 0
                gold = sets(next(iter(by_t.values())).gold)
                for t_value in schedule:
                    used += 1
                    pred = sets(by_t[float(t_value)].predicted)
                    if pred:
                        chosen = pred
                        break
                predictions.append(chosen)
                golds.append(gold)
                calls.append(used)
            if not predictions:
                continue
            metric = score(predictions, golds)
            metric.update(
                {
                    "experiment": experiment,
                    "schedule": "->".join(f"{value:.3f}" for value in schedule),
                    "scheduled_calls": length,
                    "mean_calls_with_early_stop": sum(calls) / len(calls),
                    "records": len(predictions),
                }
            )
            rows.append(metric)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-ids", required=True, help="comma-separated Slurm job IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directory-prefix", default="semantic_llm_")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    job_ids = [item.strip() for item in args.job_ids.split(",") if item.strip()]

    all_frames = []
    summaries = {}
    missing_jobs = []
    for job_id in job_ids:
        matches = list(args.root.glob(f"{args.directory_prefix}*_{job_id}"))
        if not matches:
            missing_jobs.append(job_id)
            continue
        directory = matches[0]
        if not (directory / "decisions.csv").exists():
            missing_jobs.append(job_id)
            continue
        experiment = directory.name.removeprefix(args.directory_prefix).removesuffix(f"_{job_id}")
        frame = pd.read_csv(directory / "decisions.csv", keep_default_na=False)
        frame["experiment"] = experiment
        frame["source_job"] = job_id
        all_frames.append(frame)
        summaries[experiment] = json.loads((directory / "summary.json").read_text())
    if not all_frames:
        raise RuntimeError("No completed semantic LLM outputs found")
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(args.output / "combined_decisions.csv", index=False)

    overall_rows = []
    by_slice_rows = []
    schedule_output = []
    for experiment, frame in combined.groupby("experiment"):
        meta = summaries[experiment]
        overall_rows.append({"experiment": experiment, **experiment_metrics(frame, meta)})
        for (t_value, control_type), subset in frame.groupby(["t", "control_type"]):
            row = score([sets(value) for value in subset["predicted"]], [sets(value) for value in subset["gold"]])
            by_slice_rows.append(
                {
                    "experiment": experiment,
                    "t": t_value,
                    "control_type": control_type,
                    "records": len(subset),
                    "abstain_rate": float(subset["status"].eq("abstain").mean()),
                    **row,
                }
            )
        schedule_output.extend(schedule_rows(frame, experiment))

    overall = pd.DataFrame(overall_rows).sort_values(["f1", "precision"], ascending=False)
    slices = pd.DataFrame(by_slice_rows)
    schedules = pd.DataFrame(schedule_output)
    overall.to_csv(args.output / "overall_metrics.csv", index=False)
    slices.to_csv(args.output / "metrics_by_timestep_and_type.csv", index=False)
    schedules.to_csv(args.output / "schedule_frontier.csv", index=False)
    if not schedules.empty:
        schedules.assign(
            utility=lambda x: x["f1"] - 0.02 * x["mean_calls_with_early_stop"]
        ).sort_values("utility", ascending=False).head(50).to_csv(
            args.output / "top_cost_adjusted_schedules.csv", index=False
        )
    aggregate = {
        "status": "executed",
        "jobs_requested": job_ids,
        "jobs_loaded": sorted(combined["source_job"].unique().tolist()),
        "missing_jobs": missing_jobs,
        "privacy_boundary": "Synthetic previews/counterfactuals only; no real rows sent to API.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(aggregate, indent=2))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
