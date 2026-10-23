#!/usr/bin/env python3
"""Aggregate already-computed common quality metrics for the natural DAG."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = [
    "shape_error_percent", "trend_error_percent", "alpha_precision_percent",
    "beta_recall_percent", "detection_score", "dcr_percent_50_ideal",
    "mle_xgboost_auc", "heldout_violation_error",
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

    lines = ["# Natural compositional DAG: common quality metrics", "",
             "Mean over three paired sampling seeds. Shape/trend are errors (lower is better); "
             "alpha, beta, detection and downstream AUC are higher-is-better.", "",
             "| Method | Shape ↓ | Trend ↓ | Alpha ↑ | Beta ↑ | Detection ↑ | MLE AUC ↑ | DCR |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    order = ["unconditional", "dynamic", "frozen", "late_only", "final_only"]
    for method in order:
        row = mean.loc[mean.method.eq(method)].iloc[0]
        lines.append(
            f"| {method} | {row.shape_error_percent:.4f} | {row.trend_error_percent:.4f} | "
            f"{row.alpha_precision_percent:.4f} | {row.beta_recall_percent:.4f} | "
            f"{row.detection_score:.4f} | {row.mle_xgboost_auc:.4f} | "
            f"{row.dcr_percent_50_ideal:.2f} |"
        )
    dynamic = mean.loc[mean.method.eq("dynamic")].iloc[0]
    final = mean.loc[mean.method.eq("final_only")].iloc[0]
    frozen = mean.loc[mean.method.eq("frozen")].iloc[0]
    lines += ["", "Dynamic and frozen have nearly identical general quality: absolute differences are "
              f"{dynamic.shape_error_percent-frozen.shape_error_percent:+.4f} shape, "
              f"{dynamic.trend_error_percent-frozen.trend_error_percent:+.4f} trend and "
              f"{dynamic.mle_xgboost_auc-frozen.mle_xgboost_auc:+.4f} AUC. Against final-only, "
              f"dynamic changes shape by {dynamic.shape_error_percent-final.shape_error_percent:+.4f}, "
              f"trend by {dynamic.trend_error_percent-final.trend_error_percent:+.4f} and "
              f"AUC by {dynamic.mle_xgboost_auc-final.mle_xgboost_auc:+.4f}."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
