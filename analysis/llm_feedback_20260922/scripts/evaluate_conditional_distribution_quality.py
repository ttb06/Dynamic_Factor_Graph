#!/usr/bin/env python3
"""Quality against the real held-out subset satisfying a row-level request."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_conditional_topology_controls import applicability, requirement_ok
from evaluate_shoppers_baseline_suite import alpha_beta, mle_auc, quality_and_detection


def valid(frame: pd.DataFrame, request: str):
    return (~applicability(frame)) | requirement_ok(frame, request)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--late-job-id", default="",
                        help="Optional late-reactivation array job to include.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261376)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.repo / "data/default/info.json").read_text())
    heldout = pd.read_csv(args.repo / "synthetic/default/test.csv")
    rows = []
    for request in ["range", "category", "both"]:
        dirs = {
            variant: next(args.root.glob(
                f"conditional_topology_locked_{request}_{variant}_{args.job_id}_*"
            ))
            for variant in ["dynamic", "frozen", "shuffled", "global"]
        }
        sample_specs = [
            ("unconditional", dirs["dynamic"] / "samples/unconditional.csv"),
            ("final-only", dirs["dynamic"] / "samples/final_reward_calibrated.csv"),
            ("dynamic", dirs["dynamic"] / "samples/xstate_reward_calibrated.csv"),
            ("dynamic+certifier", dirs["dynamic"] / "samples/xstate_reward_calibrated_then_final_reward.csv"),
            ("frozen", dirs["frozen"] / "samples/xstate_reward_calibrated.csv"),
            ("shuffled", dirs["shuffled"] / "samples/xstate_reward_calibrated.csv"),
            ("global", dirs["global"] / "samples/xstate_reward_calibrated.csv"),
        ]
        if args.late_job_id:
            late = next(args.root.glob(
                f"conditional_late_{request}_min.10_{args.late_job_id}_*"
            ))
            sample_specs.append((
                "dynamic+late-reactivation",
                late / "samples/xstate_reward_calibrated.csv",
            ))
        reference = heldout.loc[valid(heldout, request)].reset_index(drop=True)
        for method, path in sample_specs:
            sample = pd.read_csv(path)[heldout.columns]
            shape, trend, detection = quality_and_detection(reference, sample, info)
            alpha, beta = alpha_beta(reference, sample, info)
            rows.append({
                "request": request, "method": method, "rows": len(sample),
                "reference_rows": len(reference),
                "violation_percent": 100.0 * float((~valid(sample, request)).mean()),
                "shape_error_percent": shape, "trend_error_percent": trend,
                "alpha_precision_percent": alpha, "beta_recall_percent": beta,
                "detection_score": detection,
                "conditional_mle_xgboost_auc": mle_auc(sample, reference, info, args.seed),
                "sample_path": str(path),
            })
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "conditional_distribution_quality.csv", index=False)
    lines = [
        "# Quality against valid real held-out rows", "",
        "The reference contains real held-out rows satisfying the complete implication",
        "(`not applicable OR consequent satisfied`). This is a conditional-distribution",
        "diagnostic, not an unconditional fidelity score.", "",
        "| Request | Method | Violation | Shape | Trend | Alpha | Beta | AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.request} | {row.method} | {row.violation_percent:.3f}% | "
            f"{row.shape_error_percent:.3f} | {row.trend_error_percent:.3f} | "
            f"{row.alpha_precision_percent:.3f} | {row.beta_recall_percent:.3f} | "
            f"{row.conditional_mle_xgboost_auc:.4f} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
