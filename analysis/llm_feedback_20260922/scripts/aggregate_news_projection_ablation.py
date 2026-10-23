#!/usr/bin/env python3
"""Summarize focused public-constraint projection ablations against the fixed controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--evaluation-job", required=True)
    parser.add_argument("--control-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    labels = ["balanced_projection", "balanced_lookahead"]
    blocks = []
    control_metrics = pd.read_csv(args.control_evaluation / "metrics.csv")
    control_final = control_metrics[control_metrics.method.eq("final_repair")].copy()
    control_final["variant"] = "control_final_repair"; control_final["matched_final"] = "control_final_repair"
    control = control_metrics[control_metrics.method.eq("proposed")].copy()
    control["variant"] = "existing_controller"; control["matched_final"] = "control_final_repair"
    blocks.extend([control_final, control])
    for index, label in enumerate(labels):
        path = args.outputs / f"news_projection_eval_{args.evaluation_job}_{index}" / "metrics.csv"
        frame = pd.read_csv(path)
        final_row = frame[frame.method.eq("final_repair")].copy()
        final_row["variant"] = f"{label}_final_repair"; final_row["matched_final"] = f"{label}_final_repair"
        proposed = frame[frame.method.eq("proposed")].copy()
        proposed["variant"] = label; proposed["matched_final"] = f"{label}_final_repair"
        blocks.extend([final_row, proposed])
    result = pd.concat(blocks, ignore_index=True)
    final_lookup = result.set_index("variant")
    result["trend_gain_over_final"] = [float(row.trend_score_percent - final_lookup.loc[row.matched_final].trend_score_percent)
                                       for row in result.itertuples()]
    result["r2_gain_over_final"] = [float(row.xgboost_regression_r2 - final_lookup.loc[row.matched_final].xgboost_regression_r2)
                                    for row in result.itertuples()]
    result["nrmse_reduction_over_final"] = [float(final_lookup.loc[row.matched_final].xgboost_regression_nrmse - row.xgboost_regression_nrmse)
                                            for row in result.itertuples()]
    result["shape_gain_over_final"] = [float(row.shape_score_percent - final_lookup.loc[row.matched_final].shape_score_percent)
                                       for row in result.itertuples()]
    result.to_csv(args.output / "projection_ablation.csv", index=False)
    columns = ["variant", "shape_score_percent", "trend_score_percent", "xgboost_regression_r2",
               "xgboost_regression_nrmse", "any_constraint_violation_percent", "trend_gain_over_final",
               "r2_gain_over_final", "nrmse_reduction_over_final"]
    lines = ["# News public-constraint action projection", "",
             "All variants use the same TabDiff checkpoint, seed, dynamic trajectory gate, sample count, and final certifier.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in result[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in values) + " |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    candidates = result[result.method.eq("proposed")]
    best = candidates.sort_values(["any_constraint_violation_percent", "trend_score_percent", "xgboost_regression_r2"],
                                  ascending=[True, False, False]).iloc[0]
    (args.output / "summary.json").write_text(json.dumps({"best_variant": best["variant"],
                                                           "rows": result.to_dict("records")}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
