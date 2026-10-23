#!/usr/bin/env python3
"""Aggregate the focused external-knowledge transfer experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "shape_error_percent", "trend_error_percent", "alpha_precision_percent",
    "beta_recall_percent", "detection_score", "mle_xgboost_auc",
    "dcr_percent_50_ideal", "heldout_violation_error",
]


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    columns = [column for column in columns if column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        rendered = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reconstruction-job-id", required=True)
    parser.add_argument("--prior-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    reconstruction = []
    for path in sorted(args.root.glob(f"external_knowledge_*_{args.reconstruction_job_id}_*/metrics.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", path.parent.name.split("_")[2])
        frame["run"] = path.parent.name
        reconstruction.append(frame)
    prior = []
    for path in sorted(args.root.glob(f"semantic_prior_proxy_*_{args.prior_job_id}_*/evaluation/baseline_metrics.csv")):
        frame = pd.read_csv(path)
        parts = path.parents[1].name.split("_")
        frame.insert(0, "dataset", parts[3])
        frame.insert(1, "variant", parts[4])
        frame["run"] = path.parents[1].name
        prior.append(frame)

    recon = pd.concat(reconstruction, ignore_index=True) if reconstruction else pd.DataFrame()
    prior_frame = pd.concat(prior, ignore_index=True) if prior else pd.DataFrame()
    recon.to_csv(args.output / "reconstruction_metrics.csv", index=False)
    prior_frame.to_csv(args.output / "prior_regularization_metrics.csv", index=False)

    lines = ["# External-knowledge transfer results", ""]
    if len(recon):
        lines += [
            "## Exact-edge reconstruction and weighted repair", "",
            table(recon, ["dataset", "method", *METRICS, "row_edit_rate"]), "",
        ]
    else:
        lines += ["No reconstruction result completed successfully.", ""]
    if len(prior_frame):
        lines += [
            "## Train-time semantic-prior regularization proxy", "",
            table(prior_frame, ["dataset", "variant", "method", *METRICS]), "",
        ]
    else:
        lines += ["No prior-regularization result completed successfully.", ""]
    lines += [
        "All rows use standard distribution, detection, privacy, and TSTR metrics.",
        "These mechanisms are paper-inspired proxies and are not labeled as reproductions.",
    ]
    report = "\n".join(lines) + "\n"
    (args.output / "RESULTS.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
