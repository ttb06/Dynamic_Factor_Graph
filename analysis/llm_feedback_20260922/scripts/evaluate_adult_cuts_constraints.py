#!/usr/bin/env python3
"""Evaluate Adult samples on common metrics and public CuTS implications."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_shoppers_baseline_suite import alpha_beta, quality_and_detection, mle_auc


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype("string").str.strip()
    return result


def program_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    implication2 = ~(
        frame["marital.status"].isin(["Widowed"]) | frame["relationship"].eq("Wife")
    ) | frame["sex"].eq("Female")
    implication4 = ~frame["marital.status"].isin(["Divorced", "Never-married"]) | ~frame[
        "relationship"
    ].isin(["Husband", "Wife"])
    return {
        "implication2": implication2.to_numpy(bool),
        "implication4": implication4.to_numpy(bool),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sample", action="append", required=True, help="METHOD=csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261447)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.repo / "data/adult/info.json").read_text())
    real = clean(pd.read_csv(args.repo / "synthetic/adult/real.csv"))
    heldout = clean(pd.read_csv(args.repo / "synthetic/adult/test.csv"))
    rows = []
    for item in args.sample:
        method, path = item.split("=", 1)
        sample = clean(pd.read_csv(path))[real.columns]
        shape_error, trend_error, detection = quality_and_detection(real, sample, info)
        alpha, beta = alpha_beta(real, sample, info)
        masks = program_masks(sample)
        all_valid = np.logical_and.reduce(list(masks.values()))
        rows.append({
            "method": method,
            "rows": len(sample),
            "shape_score_percent": 100.0 - shape_error,
            "trend_score_percent": 100.0 - trend_error,
            "alpha_precision_percent": alpha,
            "beta_recall_percent": beta,
            "detection_score": detection,
            "mle_xgboost_auc": mle_auc(sample, heldout, info, args.seed),
            "implication2_satisfaction_percent": 100.0 * float(masks["implication2"].mean()),
            "implication4_satisfaction_percent": 100.0 * float(masks["implication4"].mean()),
            "all_programs_satisfaction_percent": 100.0 * float(all_valid.mean()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "metrics.csv", index=False)
    columns = list(result.columns)
    lines = [
        "# Adult comparison on public CuTS implications",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in result.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row) + " |")
    lines += [
        "",
        "Implication 2 and 4 use the public CuTS Adult program semantics.",
        "All fidelity and utility metrics are recomputed locally on the same TabDiff split.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
