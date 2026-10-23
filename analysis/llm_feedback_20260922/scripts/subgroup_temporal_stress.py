#!/usr/bin/env python3
"""Temporal-regime stress test for subgroup edges discovered on prepared splits.

This is deliberately labelled a stress test, not a clean temporal discovery
benchmark: candidate triples were discovered previously.  It recombines the
prepared rows and asks whether interaction direction persists from an early to
a late regime on datasets with an observable time proxy.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PREDICATE = re.compile(r"^\s*(.*?)\s*(==|>=|<=|>|<)\s*(.*?)\s*$")


def predicate(frame: pd.DataFrame, text: str) -> np.ndarray:
    match = PREDICATE.match(text)
    if not match: raise ValueError(text)
    column, operator, literal = (value.strip() for value in match.groups())
    if column not in frame:
        matches = [candidate for candidate in frame.columns if str(candidate).strip() == column]
        if len(matches) != 1:
            raise KeyError(column)
        column = matches[0]
    if operator == "==":
        return frame[column].astype(str).str.strip().eq(literal).to_numpy()
    values = pd.to_numeric(frame[column], errors="coerce")
    threshold = float(literal)
    return {">": values.gt, "<": values.lt, ">=": values.ge, "<=": values.le}[operator](threshold).fillna(False).to_numpy()


def split_time(dataset: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if dataset == "beijing":
        key = pd.to_datetime(dict(year=frame.year, month=frame.month, day=frame.day, hour=frame.hour), errors="coerce")
        order = np.argsort(key.to_numpy())
        description = "chronological year-month-day-hour 70/30"
    elif dataset == "news":
        # Larger timedelta means farther from the dataset collection cutoff and
        # therefore earlier publication in Online News Popularity.
        order = np.argsort(-pd.to_numeric(frame[" timedelta"], errors="coerce").to_numpy())
        description = "publication proxy timedelta descending 70/30"
    elif dataset == "shoppers":
        month_order = {"Feb": 2, "Mar": 3, "May": 5, "June": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
        key = frame.Month.astype(str).map(month_order).fillna(-1).to_numpy()
        order = np.argsort(key)
        description = "calendar month 70/30; no year available"
    else:
        raise ValueError(dataset)
    cut = int(0.70 * len(frame))
    return frame.iloc[order[:cut]].reset_index(drop=True), frame.iloc[order[cut:]].reset_index(drop=True), description


def statistics(frame: pd.DataFrame, antecedent_text: str, consequent_text: str, group_text: str) -> dict:
    a, c, g = predicate(frame, antecedent_text), predicate(frame, consequent_text), predicate(frame, group_text)
    ag, ao = a & g, a & ~g
    vg, vo = int((ag & ~c).sum()), int((ao & ~c).sum())
    ng, no = int(ag.sum()), int(ao.sum())
    rg = vg / ng if ng else math.nan
    ro = vo / no if no else math.nan
    base_g = float(c[g].mean()) if g.any() else math.nan
    base_o = float(c[~g].mean()) if (~g).any() else math.nan
    uplift_g = (1 - rg) - base_g
    uplift_o = (1 - ro) - base_o
    return {"group_support": ng, "other_support": no, "group_violation": rg, "other_violation": ro,
            "gap": rg - ro, "interaction": uplift_g - uplift_o}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cell", type=int, default=30)
    parser.add_argument("--min-interaction", type=float, default=0.025)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo / "analysis/llm_feedback_20260922/scripts"))
    import cross_dataset_constraint_audit as audit

    candidates = pd.read_csv(args.audit / "all_subgroup_edge_candidates.csv")
    candidates = candidates[candidates.dataset.isin(["beijing", "news", "shoppers"]) & candidates.train_significant].copy()
    rows = []
    split_notes = {}
    for dataset, part in candidates.groupby("dataset"):
        prepared_train, prepared_test, _, _, _ = audit.load_dataset(args.repo, dataset)
        early, late, description = split_time(dataset, pd.concat([prepared_train, prepared_test], ignore_index=True))
        split_notes[dataset] = description
        for candidate in part.itertuples(index=False):
            relation = str(candidate.constraint).split("IF ", 1)[1]
            antecedent_text, consequent_text = relation.split(" THEN ", 1)
            early_stats = statistics(early, antecedent_text, consequent_text, str(candidate.subgroup))
            late_stats = statistics(late, antecedent_text, consequent_text, str(candidate.subgroup))
            evaluable = min(early_stats["group_support"], early_stats["other_support"],
                            late_stats["group_support"], late_stats["other_support"]) >= args.min_cell
            stable = bool(evaluable and np.sign(early_stats["interaction"]) == np.sign(late_stats["interaction"])
                          and abs(late_stats["interaction"]) >= args.min_interaction)
            row = {"candidate_id": candidate.candidate_id, "dataset": dataset,
                   "constraint": candidate.constraint, "subgroup": candidate.subgroup,
                   "random_split_replicated": bool(candidate.replicated), "temporal_evaluable": evaluable,
                   "temporal_stable": stable}
            row.update({"early_" + key: value for key, value in early_stats.items()})
            row.update({"late_" + key: value for key, value in late_stats.items()})
            rows.append(row)
    result = pd.DataFrame(rows); result.to_csv(args.output / "temporal_edge_stress.csv", index=False)
    summary = result.groupby("dataset").agg(
        candidates=("candidate_id", "size"), random_replicated=("random_split_replicated", "sum"),
        temporal_evaluable=("temporal_evaluable", "sum"), temporal_stable=("temporal_stable", "sum"),
    ).reset_index()
    summary["stable_fraction_evaluable"] = summary.temporal_stable / summary.temporal_evaluable.replace(0, np.nan)
    summary.to_csv(args.output / "summary.csv", index=False)
    lines = ["# Temporal-regime stress test", "",
             "Candidates were discovered before this test, so this is not a leakage-free temporal benchmark. It is a robustness/falsification screen.", "",
             "| Dataset | Candidates | Random replicated | Temporal evaluable | Temporal stable | Stable/evaluable |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.candidates} | {row.random_replicated} | {row.temporal_evaluable} | {row.temporal_stable} | {row.stable_fraction_evaluable:.4f} |")
    lines += ["", "Split definitions:"] + [f"- {dataset}: {note}" for dataset, note in split_notes.items()]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "status": "executed", "candidate_discovery_reused": True, "clean_temporal_discovery": False,
        "purpose": "robustness stress only", "split_notes": split_notes,
    }, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__": main()
