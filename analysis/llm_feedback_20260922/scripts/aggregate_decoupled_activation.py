#!/usr/bin/env python3
"""Aggregate matched activation/proposal-source experiments across three seeds."""
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


METRICS = [
    "shape_error_percent", "trend_error_percent", "mle_xgboost_auc",
    "heldout_violation_error", "actionable_heldout_violation_error",
]
LABELS = {
    "shape_error_percent": "Shape ↓", "trend_error_percent": "Trend ↓",
    "mle_xgboost_auc": "MLE AUC ↑", "heldout_violation_error": "Held-out error ↓",
    "actionable_heldout_violation_error": "Actionable error ↓",
}


def markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def load(root: Path, pattern: str, method: str, variant: str) -> pd.DataFrame:
    files = sorted(root.glob(pattern))
    if len(files) != 6:
        raise RuntimeError(f"expected 6 files, found {len(files)} for {variant}: {pattern}")
    rows = []
    for path in files:
        dataset = "shoppers" if "_shoppers_" in str(path) else "default"
        seed = next(int(token[1:]) for token in path.parent.name.split("_") if token.startswith("s2026"))
        frame = pd.read_csv(path); selected = frame[frame.method == method]
        if len(selected) != 1: raise RuntimeError(f"missing {method} in {path}")
        row = selected.iloc[0].to_dict(); row.update(dataset=dataset, seed=seed, variant=variant)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    specs = [
        ("conditional_density_confirm_eval_*_1071210_*/baseline_metrics.csv", "final_repair", "Final repair"),
        ("global_density_matched_eval_*_1079209_*/baseline_metrics.csv", "global_density_plus_repair", "Middle / global"),
        ("global_final_matched_eval_*_1079525_*/baseline_metrics.csv", "global_final_plus_repair", "Final / global"),
        ("decoupled_activation_eval_global_*_1079601_*/baseline_metrics.csv", "decoupled_global_plus_repair", "Middle activate, final propose / global"),
        ("conditional_density_confirm_eval_*_1071210_*/baseline_metrics.csv", "conditional_density_plus_repair", "Middle / conditional"),
        ("conditional_final_matched_eval_*_1079211_*/baseline_metrics.csv", "conditional_final_plus_repair", "Final / conditional"),
        ("decoupled_activation_eval_conditional_*_1079601_*/baseline_metrics.csv", "decoupled_conditional_plus_repair", "Middle activate, final propose / conditional"),
    ]
    per_seed = pd.concat([load(a.root, *spec) for spec in specs], ignore_index=True)
    per_seed.to_csv(a.output / "per_seed_metrics.csv", index=False)
    records = []
    for (dataset, variant), group in per_seed.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": dataset, "variant": variant}
        for metric in METRICS: row[LABELS[metric]] = group[metric].mean()
        records.append(row)
    summary = pd.DataFrame(records)
    order = [spec[2] for spec in specs]
    summary["variant"] = pd.Categorical(summary.variant, order, ordered=True)
    summary = summary.sort_values(["dataset", "variant"]); summary.to_csv(a.output / "summary.csv", index=False)

    gains = []
    for dataset in sorted(per_seed.dataset.unique()):
        ref = per_seed[(per_seed.dataset == dataset) & (per_seed.variant == "Final repair")].set_index("seed")
        for variant in order[1:]:
            candidate = per_seed[(per_seed.dataset == dataset) & (per_seed.variant == variant)].set_index("seed")
            row = {"dataset": dataset, "variant": variant}
            for metric in METRICS:
                favorable = candidate[metric] - ref[metric] if metric == "mle_xgboost_auc" else ref[metric] - candidate[metric]
                row[LABELS[metric] + " favorable delta"] = favorable.mean()
                row[LABELS[metric] + " wins"] = int((favorable > 0).sum())
            for metric in ["heldout_violation_error", "actionable_heldout_violation_error"]:
                row[LABELS[metric] + " relative gain %"] = 100 * (ref[metric].mean() - candidate[metric].mean()) / ref[metric].mean()
            gains.append(row)
    gains = pd.DataFrame(gains); gains.to_csv(a.output / "gains_vs_final_repair.csv", index=False)

    show = summary.copy()
    for column in show.columns[2:]: show[column] = show[column].map(lambda x: f"{x:.6f}")
    compact = gains[["dataset", "variant", "Held-out error ↓ relative gain %",
                     "Actionable error ↓ relative gain %", "Held-out error ↓ wins", "Actionable error ↓ wins"]].copy()
    for column in compact.columns[2:4]: compact[column] = compact[column].map(lambda x: f"{x:+.2f}")
    text = "# Matched activation/proposal-source ablation\n\n"
    text += "Ba seed, 4,096 rows/seed; mọi method dùng cùng backbone, graph và evaluator độc lập.\n\n"
    text += markdown(show) + "\n\n## Semantic gains relative to final repair\n\n" + markdown(compact) + "\n"
    (a.output / "RESULTS.md").write_text(text); print(text)


if __name__ == "__main__": main()
