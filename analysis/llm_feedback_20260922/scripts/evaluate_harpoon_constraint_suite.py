#!/usr/bin/env python3
"""Evaluate exact Harpoon requests against their corresponding real subset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_shoppers_baseline_suite import alpha_beta, mle_auc, quality_and_detection


def valid(frame: pd.DataFrame, request: str) -> np.ndarray:
    age = pd.to_numeric(frame["AGE"], errors="coerce").ge(50).to_numpy()
    education = frame["EDUCATION"].astype(str).eq("3").to_numpy()
    return {"range": age, "category": education, "both": age & education}[request]


def certify(frame: pd.DataFrame, request: str) -> tuple[pd.DataFrame, float]:
    """Apply the same deterministic final certifier to any supplied baseline."""
    result = frame.copy()
    before = valid(result, request)
    if request in {"range", "both"}:
        bad = pd.to_numeric(result["AGE"], errors="coerce").lt(50)
        result.loc[bad, "AGE"] = 50
    if request in {"category", "both"}:
        bad = ~result["EDUCATION"].astype(str).eq("3")
        value = 3 if pd.api.types.is_numeric_dtype(result["EDUCATION"]) else "3"
        result.loc[bad, "EDUCATION"] = value
    return result, float(100.0 * (~before).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--request", choices=["range", "category", "both"], required=True)
    parser.add_argument("--sample", action="append", required=True, help="METHOD=/path/to/sample.csv")
    parser.add_argument("--certify-sample", action="append", default=[],
                        help="METHOD=/path/to/sample.csv; apply the shared final certifier before evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261376)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo))
    info = json.loads((args.repo / "data/default/info.json").read_text())
    heldout = pd.read_csv(args.repo / "synthetic/default/test.csv")
    reference = heldout.loc[valid(heldout, args.request)].reset_index(drop=True)
    rows = []
    items = [(item, False) for item in args.sample] + [(item, True) for item in args.certify_sample]
    for item, apply_certifier in items:
        name, path = item.split("=", 1); sample = pd.read_csv(path)[heldout.columns]
        edit_rate = float("nan")
        if apply_certifier:
            sample, edit_rate = certify(sample, args.request)
        mask = valid(sample, args.request)
        shape, trend, detection = quality_and_detection(reference, sample, info)
        alpha, beta = alpha_beta(reference, sample, info)
        rows.append({"method": name, "request": args.request, "rows": len(sample),
                     "reference_rows": len(reference), "violation_percent": 100.0 * (1-mask.mean()),
                     "shape_error_percent": shape, "trend_error_percent": trend,
                     "alpha_precision_percent": alpha, "beta_recall_percent": beta,
                     "certifier_edit_percent": edit_rate,
                     "detection_score": detection,
                     "conditional_mle_xgboost_auc": mle_auc(sample, reference, info, args.seed)})
    result = pd.DataFrame(rows); result.to_csv(args.output / "harpoon_task_metrics.csv", index=False)
    lines = [f"# Harpoon Default `{args.request}` request on TabDiff", "",
             f"Real held-out reference rows satisfying request: {len(reference)}.", "",
             "| Method | Violation % | Shape error | Trend error | Alpha precision | Beta recall | Conditional AUC |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result.itertuples(index=False):
        lines.append(f"| {row.method} | {row.violation_percent:.4f} | {row.shape_error_percent:.4f} | "
                     f"{row.trend_error_percent:.4f} | {row.alpha_precision_percent:.4f} | "
                     f"{row.beta_recall_percent:.4f} | {row.conditional_mle_xgboost_auc:.4f} |")
    lines += ["", "The constraint definitions match Harpoon; this pilot uses the repository TabDiff split/checkpoint and local evaluator, so only violation semantics—not paper fidelity values—are directly aligned."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__": main()
