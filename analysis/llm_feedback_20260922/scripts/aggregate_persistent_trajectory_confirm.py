#!/usr/bin/env python3
"""Aggregate frozen multi-seed persistent-trajectory confirmation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "shape_error_percent", "trend_error_percent", "alpha_precision_percent",
    "beta_recall_percent", "detection_score", "mle_xgboost_auc",
    "dcr_percent_50_ideal", "heldout_violation_error",
    "actionable_heldout_violation_error",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="trajectory_persistent_confirm_eval_*_{eval_job}_*/baseline_metrics.csv")
    parser.add_argument("--candidate", default="trajectory_persistent_plus_repair")
    parser.add_argument("--title", default="Persistent trajectory confirmation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    paths = sorted(args.root.glob(args.pattern.format(eval_job=args.eval_job)))
    if len(paths) != 6:
        raise RuntimeError(f"expected six evaluation tables, found {len(paths)}")
    records = []
    for path in paths:
        parts = path.parent.name.split("_")
        dataset = "shoppers" if "shoppers" in parts else "default"
        seed = int(next(value[1:] for value in parts if value.startswith("s20")))
        table = pd.read_csv(path)
        for _, row in table.iterrows():
            records.append({"dataset": dataset, "seed": seed, "method": row["method"],
                            **{metric: float(row[metric]) for metric in METRICS}})
    raw = pd.DataFrame(records); raw.to_csv(args.output / "per_seed_metrics.csv", index=False)
    summary = raw.groupby(["dataset", "method"])[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(value).rstrip("_") for value in summary.columns]
    summary.to_csv(args.output / "mean_std_metrics.csv", index=False)

    gains = []
    candidate = args.candidate
    for dataset, group in raw.groupby("dataset"):
        wide = group.pivot(index="seed", columns="method", values=METRICS)
        for metric in METRICS:
            baseline = wide[(metric, "final_repair")]
            proposed = wide[(metric, candidate)]
            lower_better = metric.endswith("error") or metric in {"shape_error_percent", "trend_error_percent"}
            delta = baseline - proposed if lower_better else proposed - baseline
            gains.append({"dataset": dataset, "metric": metric,
                          "mean_baseline": baseline.mean(), "mean_proposed": proposed.mean(),
                          "mean_favorable_delta": delta.mean(), "std_favorable_delta": delta.std(ddof=1),
                          "wins_out_of_3": int((delta > 0).sum())})
    gain_table = pd.DataFrame(gains); gain_table.to_csv(args.output / "paired_gains.csv", index=False)

    focus = gain_table[gain_table.metric.isin([
        "heldout_violation_error", "actionable_heldout_violation_error",
        "shape_error_percent", "trend_error_percent", "mle_xgboost_auc"])]
    lines = [f"# {args.title}", "",
             "Frozen policy, three unseen sampling seeds, 4,096 rows per seed.", "",
             "Positive delta always favors the proposed method.", "",
             "| Dataset | Metric | Final repair | Proposed | Favorable delta | Wins |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for _, row in focus.iterrows():
        lines.append(f"| {row.dataset} | {row.metric} | {row.mean_baseline:.6f} | "
                     f"{row.mean_proposed:.6f} | {row.mean_favorable_delta:.6f} "
                     f"± {row.std_favorable_delta:.6f} | {int(row.wins_out_of_3)}/3 |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_summary.json").write_text(json.dumps({
        "eval_job": args.eval_job, "tables": len(paths), "seeds_per_dataset": 3,
        "samples_per_seed": 4096, "policy_frozen_before_new_seeds": True}, indent=2))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
