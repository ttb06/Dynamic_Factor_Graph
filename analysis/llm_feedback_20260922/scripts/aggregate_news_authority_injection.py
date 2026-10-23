#!/usr/bin/env python3
"""Aggregate authority-separated soft-prior and soft-scorer experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "shape_score_percent", "trend_score_percent", "alpha_precision_percent",
    "beta_recall_percent", "xgboost_regression_r2", "xgboost_regression_nrmse",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for job in args.job:
        paths = sorted(args.root.glob(f"*_{job}_*"))
        for path in paths:
            metric_path = path/"eval/metrics.csv"
            if not metric_path.exists(): continue
            frame = pd.read_csv(metric_path)
            baseline = frame.loc[frame.method == "tabdiff"].iloc[0]
            guided = frame.loc[frame.method != "tabdiff"].iloc[0]
            metadata = json.loads((path/"run_metadata.json").read_text())
            trace = json.loads((path/"trace.json").read_text())
            steps = trace.get("trace", [])
            proposed = sum(int(step.get("projection_rows_proposed", 0)) for step in steps)
            rejected = sum(int(step.get("projection_rows_density_rejected", 0)) for step in steps)
            row = {"job": job, "run": path.name, "method": str(guided.method),
                   "factor_count": metadata.get("factor_count"),
                   "factor_strength": metadata.get("strength"),
                   "factor_events": trace.get("factor_cell_events", 0),
                   "projection_rows_proposed": proposed, "projection_rows_rejected": rejected,
                   "projection_reject_fraction": rejected/max(proposed, 1)}
            for metric in METRICS:
                row[metric] = guided[metric]
                delta = float(guided[metric]-baseline[metric])
                row[f"delta_{metric}"] = -delta if metric.endswith("nrmse") else delta
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty: raise RuntimeError("no completed injection evaluations found")
    result.to_csv(args.output/"metrics.csv", index=False)
    delta_columns = [column for column in result if column.startswith("delta_")]
    lines = ["# Authority-separated injection results", "",
             "Positive deltas are improvements; NRMSE is sign-flipped before reporting.", "",
             "| run | factors | strength | " + " | ".join(delta_columns) + " | reject fraction |",
             "|---|---:|---:|" + "---:|"*len(delta_columns) + "---:|"]
    for _, row in result.iterrows():
        lines.append("| " + str(row.run) + f" | {row.factor_count} | {row.factor_strength} | "
                     + " | ".join(f"{row[column]:.6f}" for column in delta_columns)
                     + f" | {row.projection_reject_fraction:.4f} |")
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(result[["run", "factor_count", "factor_strength", *delta_columns,
                  "projection_reject_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
