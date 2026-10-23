#!/usr/bin/env python3
"""Aggregate the recent-baseline campaign after all dependent arrays settle."""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


DISPLAY = {
    "shape_error_percent": "Shape error ↓",
    "trend_error_percent": "Trend error ↓",
    "alpha_precision_percent": "Alpha ↑",
    "beta_recall_percent": "Beta ↑",
    "detection_score": "Detection ↑",
    "mle_xgboost_auc": "AUC ↑",
    "dcr_percent_50_ideal": "DCR (50 ideal)",
    "actionable_heldout_violation_error": "Graph error ↓",
}


def rows_from(pattern: str, dataset: str, setting: str) -> list[dict[str, str]]:
    paths = sorted(glob.glob(pattern))
    rows: list[dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                row["dataset"] = dataset
                row["setting"] = setting
                row["source_file"] = path
                rows.append(row)
    return rows


def fmt(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return "—" if value == "" else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--forest-eval-job", required=True)
    parser.add_argument("--tabsyn-eval-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.analysis / "outputs"

    rows: list[dict[str, str]] = []
    for dataset in ("default", "shoppers"):
        rows += rows_from(
            str(root / f"forestdiffusion_eval_{dataset}_*_{args.forest_eval_job}_*" / "baseline_metrics.csv"),
            dataset,
            "local matched; K=20 screen or K=100 official",
        )
        rows += rows_from(
            str(root / f"tabsyn_official_eval_{dataset}_{args.tabsyn_eval_job}_*" / "baseline_metrics.csv"),
            dataset,
            "official code; local matched",
        )

    columns = ["dataset", "method", *DISPLAY]
    with (args.output / "recent_baseline_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*columns, "setting", "source_file"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    header = ["Dataset", "Method", *DISPLAY.values()]
    lines = ["# Recent baseline campaign", "", f"Recovered {len(rows)} evaluated rows.", "",
             "| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join([row["dataset"], row["method"], *[fmt(row, key) for key in DISPLAY]]) + " |")
    if not rows:
        lines += ["", "No evaluated row was recovered. Inspect the upstream Slurm logs; this report runs with `afterany` intentionally."]
    lines += ["", "Only the K=100 ForestDiffusion row is intended for the final main table; K=20 is a screening control.",
              "All displayed rows use the same local split, 4,096-row sample size, and evaluator as the proposed method."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
