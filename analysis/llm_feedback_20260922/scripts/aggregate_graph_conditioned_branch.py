#!/usr/bin/env python3
"""Aggregate the new graph-conditioned method branch after Slurm evaluation."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.5f}")
    return "\n".join([
        "| " + " | ".join(map(str, shown.columns)) + " |",
        "| " + " | ".join(["---"] * len(shown.columns)) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--experiment", action="append", required=True, help="LABEL=SLURM_EVAL_JOB")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for specification in args.experiment:
        label, job = specification.split("=", 1)
        for path in sorted(args.outputs.glob(f"*_{job}_*/baseline_metrics.csv")):
            dirname = path.parent.name
            dataset = "shoppers" if "shoppers" in dirname else "diabetes" if "diabetes" in dirname else "unknown"
            match = re.search(rf"{dataset}_(.+?)(?:_s\d+)?_{job}_\d+$", dirname)
            variant = match.group(1) if match else dirname
            frame = pd.read_csv(path)
            base = frame.set_index("method")
            if "final_repair" not in base.index or "tabdiff" not in base.index: continue
            final = base.loc["final_repair"]; tabdiff = base.loc["tabdiff"]
            for _, row in frame.iterrows():
                record = {"experiment": label, "dataset": dataset, "variant": variant,
                          "run": dirname, **row.to_dict()}
                if row.method in {"graph_conditioned", "graph_conditioned_plus_repair"}:
                    record["heldout_gain_vs_tabdiff_pct"] = 100 * (tabdiff.heldout_violation_error - row.heldout_violation_error) / max(tabdiff.heldout_violation_error, 1e-12)
                    record["heldout_gain_vs_final_pct"] = 100 * (final.heldout_violation_error - row.heldout_violation_error) / max(final.heldout_violation_error, 1e-12)
                    record["shape_gain_vs_final_pct"] = 100 * (final.shape_error_percent - row.shape_error_percent) / max(final.shape_error_percent, 1e-12)
                    record["trend_gain_vs_final_pct"] = 100 * (final.trend_error_percent - row.trend_error_percent) / max(final.trend_error_percent, 1e-12)
                    record["auc_delta_vs_final"] = row.mle_xgboost_auc - final.mle_xgboost_auc
                    record["pareto_vs_final"] = bool(
                        row.heldout_violation_error <= final.heldout_violation_error
                        and row.shape_error_percent <= final.shape_error_percent
                        and row.trend_error_percent <= final.trend_error_percent
                    )
                rows.append(record)
    full = pd.DataFrame(rows); full.to_csv(args.output / "all_results.csv", index=False)
    methods = full[full.method.isin(["graph_conditioned", "graph_conditioned_plus_repair"])].copy()
    metrics = ["heldout_violation_error", "shape_error_percent", "trend_error_percent",
               "alpha_precision_percent", "mle_xgboost_auc", "heldout_gain_vs_tabdiff_pct",
               "heldout_gain_vs_final_pct", "shape_gain_vs_final_pct", "trend_gain_vs_final_pct",
               "auc_delta_vs_final"]
    if len(methods):
        summary = methods.groupby(["experiment", "dataset", "variant", "method"], as_index=False)[metrics].agg(["mean", "std"])
        summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
        summary.to_csv(args.output / "summary.csv", index=False)
        display_columns = ["experiment", "dataset", "variant", "method",
                           "heldout_violation_error_mean", "shape_error_percent_mean",
                           "trend_error_percent_mean", "mle_xgboost_auc_mean",
                           "heldout_gain_vs_tabdiff_pct_mean", "heldout_gain_vs_final_pct_mean"]
        report_table = summary[[column for column in display_columns if column in summary]]
    else:
        summary = pd.DataFrame(); report_table = summary
    metadata = {"experiments": args.experiment, "runs_loaded": int(full.run.nunique()) if len(full) else 0,
                "rows": len(full), "selection": "none; descriptive aggregation of all submitted runs"}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    lines = ["# Graph-conditioned method branch", "", table(report_table) if len(report_table) else "No completed runs.",
             "", "Positive gain values mean lower error than the named baseline. Paper claims require paired multi-seed confirmation and common-metric Pareto checks."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__": main()
