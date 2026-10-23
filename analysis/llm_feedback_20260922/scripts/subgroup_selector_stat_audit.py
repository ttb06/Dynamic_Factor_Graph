#!/usr/bin/env python3
"""Post-hoc verification and clustered uncertainty for subgroup selectors."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ACTIVE = {"activate_in_group", "deactivate_in_group_as_exception"}


def scores(gold: np.ndarray, selected: np.ndarray) -> tuple[float, float, float]:
    tp = int((gold & selected).sum())
    fp = int((~gold & selected).sum())
    fn = int((gold & ~selected).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and np.isfinite(precision + recall) else np.nan
    return precision, recall, f1


def selections(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    conservative = frame.conservative_action.isin(ACTIVE).to_numpy()
    statistics_verified = (
        frame.statistics_action.isin(ACTIVE) & ~frame.sensitive_or_demographic.astype(bool)
    ).to_numpy()
    raw = frame.action.isin(ACTIVE)
    direction = (
        (frame.action.eq("activate_in_group") & frame.graph_interpretation.eq("edge_active_in_subgroup"))
        | (frame.action.eq("deactivate_in_group_as_exception") & frame.graph_interpretation.eq("exception_subgroup"))
    )
    verified = (raw & ~frame.sensitive_or_demographic.astype(bool) & direction).to_numpy()
    confidence = pd.to_numeric(frame.confidence, errors="coerce").fillna(-1).to_numpy()
    return {
        "verified_statistics_all": statistics_verified,
        "conservative_statistics": conservative,
        "verified_llm": verified,
        "verified_llm_confidence_0.8": verified & (confidence >= 0.8),
        "verified_llm_confidence_0.9": verified & (confidence >= 0.9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    summary_rows, comparison_rows, per_dataset = [], [], []
    for run in args.runs:
        frame = pd.read_csv(run / "evaluated_decisions.csv")
        model = str(frame.get("model", pd.Series([run.name])).iloc[0]) if "model" in frame else run.name
        if model == run.name:
            metadata_path = run / "run_metadata.json"
            if metadata_path.exists():
                import json
                model = json.loads(metadata_path.read_text()).get("model", run.name)
        gold = (frame.replicated.astype(bool) & ~frame.sensitive_or_demographic.astype(bool)).to_numpy()
        methods = selections(frame)
        for method, selected in methods.items():
            precision, recall, f1 = scores(gold, selected)
            summary_rows.append({"model": model, "method": method, "candidates": len(frame),
                                 "selected": int(selected.sum()), "precision": precision,
                                 "recall": recall, "f1": f1})
            for dataset, indices in frame.groupby("dataset").groups.items():
                idx = np.asarray(list(indices), dtype=int)
                p, r, f = scores(gold[idx], selected[idx])
                per_dataset.append({"model": model, "method": method, "dataset": dataset,
                                    "candidates": len(idx), "selected": int(selected[idx].sum()),
                                    "precision": p, "recall": r, "f1": f})

        cluster = (frame.dataset.astype(str) + "|" + frame.constraint_id.astype(str)).to_numpy()
        unique = np.unique(cluster)
        cluster_indices = {key: np.flatnonzero(cluster == key) for key in unique}
        for baseline in ["verified_statistics_all", "conservative_statistics"]:
            for method in ["verified_llm", "verified_llm_confidence_0.8", "verified_llm_confidence_0.9"]:
                deltas = []
                for _ in range(args.bootstrap):
                    sampled = rng.choice(unique, len(unique), replace=True)
                    idx = np.concatenate([cluster_indices[key] for key in sampled])
                    p0, r0, f0 = scores(gold[idx], methods[baseline][idx])
                    p1, r1, f1 = scores(gold[idx], methods[method][idx])
                    deltas.append((p1 - p0, r1 - r0, f1 - f0))
                values = np.asarray(deltas)
                for ordinal, metric in enumerate(["precision", "recall", "f1"]):
                    comparison_rows.append({
                        "model": model, "method": method, "baseline": baseline, "metric": metric,
                        "delta": float(np.nanmean(values[:, ordinal])),
                        "ci95_low": float(np.nanquantile(values[:, ordinal], 0.025)),
                        "ci95_high": float(np.nanquantile(values[:, ordinal], 0.975)),
                        "probability_delta_positive": float(np.nanmean(values[:, ordinal] > 0)),
                        "bootstrap_clusters": len(unique), "bootstrap_replicates": args.bootstrap,
                    })
    summary = pd.DataFrame(summary_rows)
    comparisons = pd.DataFrame(comparison_rows)
    summary.to_csv(args.output / "verified_summary.csv", index=False)
    comparisons.to_csv(args.output / "cluster_bootstrap.csv", index=False)
    pd.DataFrame(per_dataset).to_csv(args.output / "per_dataset.csv", index=False)
    lines = ["# Verified subgroup-selector audit", "",
             "Verification blocks sensitive candidates and LLM actions whose direction contradicts the train interaction. Confidence thresholds were evaluated at the pre-specified values 0.8 and 0.9. Recall is within the API payload, not over the full candidate bank.", "",
             "## Point estimates", "",
             "| Model | Method | Selected | Precision | Recall | F1 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.model} | {row.method} | {row.selected} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} |")
    lines += ["", "## Cluster bootstrap versus verified statistical baselines", "",
              "| Model | Method | Baseline | Metric | Delta | 95% interval | P(delta>0) |",
              "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for row in comparisons.itertuples(index=False):
        lines.append(f"| {row.model} | {row.method} | {row.baseline} | {row.metric} | {row.delta:.4f} | [{row.ci95_low:.4f}, {row.ci95_high:.4f}] | {row.probability_delta_positive:.3f} |")
    lines += ["", "Candidates sharing a dataset and original constraint edge are resampled as one cluster. The audit remains exploratory because the payload is top-ranked and heldout is mostly a random split."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
