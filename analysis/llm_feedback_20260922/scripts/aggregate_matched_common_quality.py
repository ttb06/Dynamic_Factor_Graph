#!/usr/bin/env python3
"""Aggregate matched common-quality tables and quantify deltas from TabDiff."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = [
    "shape_error_percent", "trend_error_percent", "alpha_precision_percent",
    "beta_recall_percent", "detection_score", "mle_xgboost_auc",
    "dcr_percent_50_ideal", "heldout_violation_error",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    frames = []
    for path in args.inputs:
        match = re.search(r"_s(\d+)_", str(path))
        if not match:
            raise ValueError(f"cannot parse seed from {path}")
        frame = pd.read_csv(path); frame["seed"] = int(match.group(1)); frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(args.output / "per_seed_quality.csv", index=False)
    mean = raw.groupby("method")[METRICS].mean().reset_index()
    std = raw.groupby("method")[METRICS].std(ddof=1).reset_index()
    mean.to_csv(args.output / "mean_quality.csv", index=False)
    std.to_csv(args.output / "std_quality.csv", index=False)

    base = mean.loc[mean.method.eq("tabdiff")].iloc[0]
    deltas = mean.copy()
    for metric in METRICS:
        deltas[f"{metric}_delta_vs_tabdiff"] = deltas[metric] - float(base[metric])
    deltas.to_csv(args.output / "deltas_vs_tabdiff.csv", index=False)

    order = ["tabdiff", "dynamic_graph", "frozen_graph", "final_repair",
             "harpoon_hard", "harpoon_calibrated"]
    lines = ["# Common metrics: TabDiff, dynamic graph and Harpoon-style proxy", "",
             "Mean over three paired sampling seeds. Shape/trend and held-out error are lower-is-better; "
             "alpha, beta, detection and downstream AUC are higher-is-better. DCR is best interpreted "
             "by distance to 50 after equalizing train/test reference pool sizes.", "",
             "| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | AUC ↑ | DCR (50 ideal) | Held-out error ↓ |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in order:
        row = mean.loc[mean.method.eq(method)].iloc[0]
        lines.append(f"| {method} | {row.shape_error_percent:.4f} | {row.trend_error_percent:.4f} | "
                     f"{row.alpha_precision_percent:.4f} | {row.beta_recall_percent:.4f} | "
                     f"{row.detection_score:.4f} | {row.mle_xgboost_auc:.4f} | "
                     f"{row.dcr_percent_50_ideal:.2f} | {row.heldout_violation_error:.4f} |")
    lines += ["", "Absolute deltas from TabDiff:", "",
              "| Method | ΔShape | ΔTrend | ΔAlpha | ΔBeta | ΔDetection | ΔAUC | Held-out gain |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in order[1:]:
        row = mean.loc[mean.method.eq(method)].iloc[0]
        heldout_gain = 100.0 * (float(base.heldout_violation_error) - float(row.heldout_violation_error)) / max(float(base.heldout_violation_error), 1e-12)
        lines.append(f"| {method} | {row.shape_error_percent-base.shape_error_percent:+.4f} | "
                     f"{row.trend_error_percent-base.trend_error_percent:+.4f} | "
                     f"{row.alpha_precision_percent-base.alpha_precision_percent:+.4f} | "
                     f"{row.beta_recall_percent-base.beta_recall_percent:+.4f} | "
                     f"{row.detection_score-base.detection_score:+.4f} | "
                     f"{row.mle_xgboost_auc-base.mle_xgboost_auc:+.4f} | {heldout_gain:+.1f}% |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
