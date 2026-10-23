#!/usr/bin/env python3
"""Aggregate matched News public-constraint runs over fixed confirmation seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DIRECTIONS = {
    "shape_score_percent": 1,
    "trend_score_percent": 1,
    "alpha_precision_percent": 1,
    "beta_recall_percent": 1,
    "detection_score": 1,
    "xgboost_regression_r2": 1,
    "xgboost_regression_nrmse": -1,
    "any_constraint_violation_percent": -1,
}


def markdown(frame: pd.DataFrame) -> str:
    header = list(frame.columns)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames = []
    for index, directory in enumerate(args.evaluation):
        frame = pd.read_csv(directory / "metrics.csv")
        frame.insert(0, "run", index)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(args.output / "per_seed_metrics.csv", index=False)
    metrics = [metric for metric in DIRECTIONS if metric in raw]
    summary = raw.groupby("method")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "method_summary.csv", index=False)

    paired_rows = []
    for metric in metrics:
        pivot = raw.pivot(index="run", columns="method", values=metric)
        delta = pivot["proposed"] - pivot["final_repair"]
        oriented = DIRECTIONS[metric] * delta
        paired_rows.append({
            "metric": metric,
            "proposed_minus_final_mean": float(delta.mean()),
            "proposed_minus_final_std": float(delta.std()),
            "wins": int((oriented > 0).sum()),
            "ties": int((oriented == 0).sum()),
            "seeds": len(delta),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.output / "paired_deltas.csv", index=False)
    lines = [
        "# News public-constraint confirmation",
        "",
        "All runs use the same checkpoint, public graph, controller policy and sample count; only sampling seed changes.",
        "",
        "## Mean and standard deviation",
        "",
        markdown(summary),
        "",
        "## Proposed versus matched final repair",
        "",
        markdown(paired),
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
