#!/usr/bin/env python3
"""Aggregate completed News end-to-end frontier tasks against paired e4000 TabDiff."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "shape_score_percent", "trend_score_percent", "alpha_precision_percent",
    "beta_recall_percent", "xgboost_regression_r2", "xgboost_regression_nrmse",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    reference_frame = pd.read_csv(args.reference)
    reference = reference_frame.loc[
        reference_frame["method"].astype(str).str.startswith("tabdiff")].iloc[0]
    rows = []
    patterns = ("news_e2e_teacher_*", "news_e2e_strength_*", "news_e2e_fullref_*",
                "news_e2e_confirm_*", "news_e2e_ablation_*",
                "news_e2e_resample_*", "news_e2e_benefit_*",
                "news_benefit_route_*", "news_learned_gate_*",
                "news_hybrid_gate_*")
    visited = set()
    for pattern in patterns:
        for directory in sorted(args.outputs.glob(pattern)):
            if directory in visited:
                continue
            visited.add(directory)
            metrics_path = directory/"eval"/"metrics.csv"
            if not metrics_path.exists():
                metrics_path = directory/"eval_recovered"/"metrics.csv"
            config_path = directory/"train"/"experiment_config.json"
            if not metrics_path.exists():
                continue
            try:
                frame = pd.read_csv(metrics_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                # Interrupted Slurm tasks can leave an empty evaluator file;
                # absence of a complete table is not an experimental result.
                continue
            for _, record in frame.iterrows():
                row = {"run": directory.name, "method": record["method"]}
                for metric in METRICS:
                    row[metric] = record[metric]
                    direction = -1.0 if metric.endswith("nrmse") else 1.0
                    row[f"delta_{metric}"] = direction*(record[metric]-reference[metric])
                row["config"] = str(config_path) if config_path.exists() else ""
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(args.output/"results.csv", index=False)
    if result.empty:
        (args.output/"RESULTS.md").write_text("No completed result was found.\n")
        return
    columns = ["method", *METRICS, *[f"delta_{metric}" for metric in METRICS]]
    table = result[columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"]*len(columns)) + "|"
    body = []
    for _, record in table.iterrows():
        values = [str(record[column]) if column == "method" else f"{float(record[column]):.6f}"
                  for column in columns]
        body.append("| " + " | ".join(values) + " |")
    markdown = ["# News end-to-end trajectory-teacher frontier", "",
                "Deltas use higher-is-better orientation; NRMSE delta is negated.", "",
                header, separator, *body, ""]
    (args.output/"RESULTS.md").write_text("\n".join(markdown))
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
