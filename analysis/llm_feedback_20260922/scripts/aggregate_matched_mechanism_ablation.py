#!/usr/bin/env python3
"""Aggregate the matched selector and trigger-state ablations."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = {
    "shape_error_percent": "Shape ↓",
    "trend_error_percent": "Trend ↓",
    "mle_xgboost_auc": "MLE AUC ↑",
    "heldout_violation_error": "Held-out error ↓",
    "actionable_heldout_violation_error": "Actionable error ↓",
}


def markdown(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table writer."""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def load(root: Path, pattern: str, job: str, method: str, label: str) -> pd.DataFrame:
    files = sorted(root.glob(pattern.format(job=job)))
    if len(files) != 6:
        raise RuntimeError(f"expected six files for {label}, found {len(files)}: {pattern}")
    rows = []
    for path in files:
        dataset = "shoppers" if "_shoppers_" in str(path) else "default"
        frame = pd.read_csv(path)
        selected = frame[frame.method == method]
        if len(selected) != 1:
            raise RuntimeError(f"method {method} absent or duplicated in {path}")
        row = selected.iloc[0].to_dict(); row.update(dataset=dataset, variant=label, source=str(path))
        rows.append(row)
    return pd.DataFrame(rows)


def favorable_delta(metric: str, candidate: float, baseline: float) -> float:
    return candidate - baseline if metric == "mle_xgboost_auc" else baseline - candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mid-eval-job", required=True)
    parser.add_argument("--global-eval-job", required=True)
    parser.add_argument("--final-eval-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    tables = [
        load(args.root, "conditional_density_confirm_eval_*_{job}_*/baseline_metrics.csv",
             args.mid_eval_job, "final_repair", "Final repair"),
        load(args.root, "global_density_matched_eval_*_{job}_*/baseline_metrics.csv",
             args.global_eval_job, "global_density_plus_repair", "Middle + global selector"),
        load(args.root, "conditional_final_matched_eval_*_{job}_*/baseline_metrics.csv",
             args.final_eval_job, "conditional_final_plus_repair", "Final + conditional selector"),
        load(args.root, "conditional_density_confirm_eval_*_{job}_*/baseline_metrics.csv",
             args.mid_eval_job, "conditional_density_plus_repair", "Middle + conditional selector"),
    ]
    per_seed = pd.concat(tables, ignore_index=True)
    per_seed.to_csv(args.output / "per_seed_metrics.csv", index=False)

    records = []
    for (dataset, variant), group in per_seed.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": dataset, "variant": variant}
        for metric, label in METRICS.items():
            row[label] = group[metric].mean(); row[label + " std"] = group[metric].std(ddof=1)
        records.append(row)
    summary = pd.DataFrame(records)
    order = ["Final repair", "Middle + global selector", "Final + conditional selector",
             "Middle + conditional selector"]
    summary["variant"] = pd.Categorical(summary.variant, order, ordered=True)
    summary = summary.sort_values(["dataset", "variant"])
    summary.to_csv(args.output / "summary.csv", index=False)

    deltas = []
    for dataset in sorted(per_seed.dataset.unique()):
        subset = per_seed[per_seed.dataset == dataset]
        means = subset.groupby("variant").mean(numeric_only=True)
        for comparison, baseline in [
            ("conditional score vs global score", "Middle + global selector"),
            ("middle trigger vs final trigger", "Final + conditional selector"),
            ("full method vs final repair", "Final repair"),
        ]:
            candidate = "Middle + conditional selector"
            row = {"dataset": dataset, "comparison": comparison}
            for metric, label in METRICS.items():
                row[label + " favorable delta"] = favorable_delta(
                    metric, means.loc[candidate, metric], means.loc[baseline, metric])
            deltas.append(row)
    delta_frame = pd.DataFrame(deltas); delta_frame.to_csv(args.output / "favorable_deltas.csv", index=False)

    display = summary.copy()
    for column in display.columns[2:]: display[column] = display[column].map(lambda value: f"{value:.6f}")
    delta_display = delta_frame.copy()
    for column in delta_display.columns[2:]: delta_display[column] = delta_display[column].map(lambda value: f"{value:+.6f}")
    text = "# Matched mechanism ablation\n\n"
    text += "Ba seed, 4,096 rows/seed. Positive favorable delta means the full middle-conditional method is better.\n\n"
    text += markdown(display) + "\n\n## Direct favorable deltas\n\n"
    text += markdown(delta_display) + "\n"
    (args.output / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
