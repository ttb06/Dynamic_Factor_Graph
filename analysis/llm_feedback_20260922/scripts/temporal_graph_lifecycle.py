#!/usr/bin/env python3
"""Leakage-free temporal discovery and graph-confidence update audit.

Candidate predicates and implications are created using the early regime only.
The middle regime is available to graph-update policies, while the late regime
is held out for evaluation.  This tests whether a constraint graph should be
frozen after discovery or updated when its empirical reliability drifts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def chronological(dataset: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if dataset == "beijing":
        key = pd.to_datetime(
            dict(year=frame.year, month=frame.month, day=frame.day, hour=frame.hour),
            errors="coerce",
        )
        order = np.argsort(key.to_numpy())
        note = "year-month-day-hour"
    elif dataset == "news":
        # A larger timedelta is farther from collection cutoff and hence older.
        order = np.argsort(-pd.to_numeric(frame[" timedelta"], errors="coerce").to_numpy())
        note = "timedelta descending publication proxy"
    elif dataset == "shoppers":
        month_order = {
            "Feb": 2, "Mar": 3, "May": 5, "June": 6, "Jul": 7,
            "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }
        order = np.argsort(frame.Month.astype(str).map(month_order).fillna(-1).to_numpy())
        note = "calendar month; no year available"
    else:
        raise ValueError(dataset)
    return frame.iloc[order].reset_index(drop=True), note


def metrics(applicable: np.ndarray, satisfied: np.ndarray) -> tuple[int, int, float]:
    n = int(applicable.sum())
    bad = int((applicable & ~satisfied).sum())
    return n, bad, bad / n if n else math.nan


def wilson_high(bad: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return math.nan
    p = bad / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / den
    return min(1.0, centre + radius)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["beijing", "news", "shoppers"])
    p.add_argument("--min-support", type=int, default=100)
    p.add_argument("--min-confidence", type=float, default=0.80)
    p.add_argument("--max-per-antecedent", type=int, default=12)
    p.add_argument("--hard-threshold", type=float, default=0.05)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.repo / "analysis/llm_feedback_20260922/scripts"))
    import cross_dataset_constraint_audit as audit

    rows: list[dict] = []
    notes: dict[str, str] = {}
    for dataset in a.datasets:
        train, test, categorical, numerical, target = audit.load_dataset(a.repo, dataset)
        ordered, note = chronological(dataset, pd.concat([train, test], ignore_index=True))
        n = len(ordered)
        early = ordered.iloc[: int(0.50 * n)].reset_index(drop=True)
        middle = ordered.iloc[int(0.50 * n): int(0.70 * n)].reset_index(drop=True)
        late = ordered.iloc[int(0.70 * n):].reset_index(drop=True)
        future = pd.concat([middle, late], ignore_index=True)
        notes[dataset] = note
        predicates = audit.predicate_bank(early, future, categorical, numerical, a.min_support)
        prevalence = {x.predicate_id: float(x.train_mask.mean()) for x in predicates}
        cut = len(middle)
        for antecedent in predicates:
            candidates = []
            for consequent in predicates:
                if antecedent.column == consequent.column:
                    continue
                e_app, e_bad, e_rate = metrics(antecedent.train_mask, consequent.train_mask)
                if not np.isfinite(e_rate) or 1 - e_rate < a.min_confidence:
                    continue
                lift = (1 - e_rate) / prevalence[consequent.predicate_id] if prevalence[consequent.predicate_id] else math.inf
                if lift < 1.10 and e_rate > 0.01:
                    continue
                m_app, m_bad, m_rate = metrics(antecedent.test_mask[:cut], consequent.test_mask[:cut])
                l_app, l_bad, l_rate = metrics(antecedent.test_mask[cut:], consequent.test_mask[cut:])
                if min(e_app, m_app, l_app) < 30:
                    continue
                candidates.append({
                    "dataset": dataset,
                    "edge": f"IF {antecedent.description} THEN {consequent.description}",
                    "antecedent_column": antecedent.column,
                    "consequent_column": consequent.column,
                    "sensitive_or_demographic": audit.is_sensitive([antecedent.column, consequent.column]),
                    "target_related": bool(target and target in [antecedent.column, consequent.column]),
                    "early_support": e_app, "middle_support": m_app, "late_support": l_app,
                    "early_bad": e_bad, "middle_bad": m_bad, "late_bad": l_bad,
                    "early_rate": e_rate, "middle_rate": m_rate, "late_rate": l_rate,
                    "early_wilson_high": wilson_high(e_bad, e_app),
                    "middle_wilson_high": wilson_high(m_bad, m_app),
                    "early_lift": lift,
                })
            candidates.sort(key=lambda x: (x["early_bad"] > 0, x["early_rate"], -min(x["early_lift"], 100), -x["early_support"]))
            rows.extend(candidates[: a.max_per_antecedent])

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no temporal graph edges passed discovery thresholds")
    safe = ~frame.sensitive_or_demographic & ~frame.target_related
    frame["late_stable"] = (
        frame.late_rate.le(a.hard_threshold)
        & frame.late_rate.sub(frame.early_rate).abs().le(a.hard_threshold)
    )
    # Update decisions use no late labels.
    frame["frozen_keep"] = safe & frame.early_rate.le(a.hard_threshold)
    frame["middle_point_keep"] = frame.frozen_keep & frame.middle_rate.le(a.hard_threshold)
    frame["middle_wilson_keep"] = frame.frozen_keep & frame.middle_wilson_high.le(a.hard_threshold)
    # A predicate bank can emit several threshold/value variants for the same
    # ordered column pair. Select one representative using early evidence only
    # to expose how much of an apparent shift is duplicated graph structure.
    frame["pair_representative"] = False
    eligible = frame[frame.frozen_keep].sort_values(
        ["dataset", "antecedent_column", "consequent_column", "early_wilson_high", "early_support"],
        ascending=[True, True, True, True, False],
    )
    representative_indices = eligible.groupby(
        ["dataset", "antecedent_column", "consequent_column"], sort=False
    ).head(1).index
    frame.loc[representative_indices, "pair_representative"] = True
    frame["frozen_pair_dedup_keep"] = frame.frozen_keep & frame.pair_representative
    frame["middle_point_pair_dedup_keep"] = frame.middle_point_keep & frame.pair_representative
    frame["middle_wilson_pair_dedup_keep"] = frame.middle_wilson_keep & frame.pair_representative
    frame["early_prediction_abs_error"] = (frame.early_rate - frame.late_rate).abs()
    frame["middle_prediction_abs_error"] = (frame.middle_rate - frame.late_rate).abs()
    pooled_bad = frame.early_bad + frame.middle_bad
    pooled_n = frame.early_support + frame.middle_support
    frame["pooled_rate"] = pooled_bad / pooled_n
    frame["pooled_prediction_abs_error"] = (frame.pooled_rate - frame.late_rate).abs()
    frame.to_csv(a.output / "temporal_edges.csv", index=False)

    summary = []
    for dataset, g in frame.groupby("dataset"):
        f = g[g.frozen_keep]
        for policy in [
            "frozen_keep", "middle_point_keep", "middle_wilson_keep",
            "frozen_pair_dedup_keep", "middle_point_pair_dedup_keep",
            "middle_wilson_pair_dedup_keep",
        ]:
            universe = f[f.pair_representative] if "pair_dedup" in policy else f
            selected = universe[universe[policy]] if policy != "frozen_keep" else universe
            tp = int(selected.late_stable.sum())
            predicted = len(selected)
            positives = int(universe.late_stable.sum())
            summary.append({
                "dataset": dataset, "policy": policy,
                "early_discovered_hard_edges": len(universe), "selected": predicted,
                "stable_selected": tp,
                "late_stability_precision": tp / predicted if predicted else math.nan,
                "late_stability_recall": tp / positives if positives else math.nan,
                "unsafe_late_activations": predicted - tp,
                "weighted_late_violation": float(np.average(selected.late_rate, weights=selected.late_support)) if predicted else math.nan,
                "early_rate_mae": float(f.early_prediction_abs_error.mean()) if len(f) else math.nan,
                "middle_rate_mae": float(f.middle_prediction_abs_error.mean()) if len(f) else math.nan,
                "pooled_rate_mae": float(f.pooled_prediction_abs_error.mean()) if len(f) else math.nan,
            })
    out = pd.DataFrame(summary)
    out.to_csv(a.output / "summary.csv", index=False)
    lines = [
        "# Leakage-free temporal graph lifecycle audit", "",
        "Edges and thresholds are discovered on the first 50% only. The next 20% may update the graph; the final 30% is held out.", "",
        "| Dataset | Update policy | Early hard edges | Selected | Late-stable | Precision | Recall | Unsafe late activations | Weighted late violation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in out.itertuples(index=False):
        lines.append(
            f"| {r.dataset} | {r.policy} | {r.early_discovered_hard_edges} | {r.selected} | {r.stable_selected} | "
            f"{r.late_stability_precision:.4f} | {r.late_stability_recall:.4f} | {r.unsafe_late_activations} | {r.weighted_late_violation:.4f} |"
        )
    lines += ["", "Time proxies:"] + [f"- {k}: {v}" for k, v in notes.items()]
    (a.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (a.output / "run_metadata.json").write_text(json.dumps({
        "status": "executed", "discovery": "early 50% only", "update": "middle 20% only",
        "evaluation": "late 30% only", "hard_threshold": a.hard_threshold,
        "semantic_claim": False, "purpose": "graph reliability lifecycle and shift falsification",
    }, indent=2))
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
