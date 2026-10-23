#!/usr/bin/env python3
"""Audit whether every atom in an LLM-proposed hyperedge is empirically necessary.

This is a training-only structural audit.  It never makes a semantic rule true;
it only rejects inflated conjunctions whose apparent complexity is unsupported.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from knowledge_graph_protocol import predicate_mask


def masks(frame: pd.DataFrame, edge: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.ones(len(frame), dtype=bool)
    for atom in edge.get("applicability", []):
        active &= predicate_mask(frame, atom)
    excluded = np.zeros(len(frame), dtype=bool)
    for group in edge.get("exceptions", []):
        group_mask = np.ones(len(frame), dtype=bool)
        for atom in group.get("all_of", []):
            group_mask &= predicate_mask(frame, atom)
        excluded |= group_mask
    active &= ~excluded
    satisfied = np.ones(len(frame), dtype=bool)
    for atom in edge.get("requirements", []):
        satisfied &= predicate_mask(frame, atom)
    return active, satisfied, active & ~satisfied


def stats(frame: pd.DataFrame, edge: dict) -> dict:
    active, _, violation = masks(frame, edge)
    support = int(active.sum()); violations = int(violation.sum())
    return {"support": support, "violations": violations,
            "violation_rate": violations / support if support else np.nan}


def atom_text(atom: dict) -> str:
    right = atom.get("right_column") if atom.get("right_kind") == "column" else repr(atom.get("right_value"))
    return f"{atom.get('left_column')} {atom.get('operator')} {right}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="shoppers")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261222)
    parser.add_argument("--min-new-support", type=int, default=30)
    parser.add_argument("--min-rate-increase", type=float, default=.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(args.repo / "analysis/llm_feedback_20260922/scripts"))
    import cross_dataset_constraint_audit as audit
    train, _, categorical, numerical, target = audit.load_dataset(args.repo, args.dataset)
    rng = np.random.default_rng(args.seed); order = rng.permutation(len(train)); n = len(train)
    splits = {
        "discovery": train.iloc[order[:int(.60*n)]].copy(),
        "counterexample": train.iloc[order[int(.60*n):int(.80*n)]].copy(),
        "calibration": train.iloc[order[int(.80*n):]].copy(),
    }
    edges = json.loads(args.graph.read_text())
    atom_rows, edge_rows = [], []
    for edge in edges:
        if len({atom.get("left_column") for atom in edge.get("applicability", []) + edge.get("requirements", [])}) < 3:
            continue
        app_passes = []
        for index, atom in enumerate(edge.get("applicability", [])):
            split_passes = 0
            for split_name, frame in splits.items():
                base = stats(frame, edge)
                reduced = copy.deepcopy(edge); reduced["applicability"].pop(index)
                relaxed = stats(frame, reduced)
                base_active, _, _ = masks(frame, edge)
                relaxed_active, _, relaxed_bad = masks(frame, reduced)
                new_region = relaxed_active & ~base_active
                new_support = int(new_region.sum()); new_bad = int((relaxed_bad & new_region).sum())
                rate_increase = relaxed["violation_rate"] - base["violation_rate"]
                necessary = (new_support >= args.min_new_support and
                             ((np.isfinite(rate_increase) and rate_increase >= args.min_rate_increase)
                              or new_bad >= 5))
                split_passes += int(necessary)
                atom_rows.append({
                    "candidate_id": edge["candidate_id"], "atom_type": "applicability",
                    "atom_index": index, "atom": atom_text(atom), "split": split_name,
                    "base_support": base["support"], "relaxed_support": relaxed["support"],
                    "new_region_support": new_support, "new_region_violations": new_bad,
                    "base_violation_rate": base["violation_rate"],
                    "relaxed_violation_rate": relaxed["violation_rate"],
                    "rate_increase": rate_increase, "necessary_on_split": necessary,
                })
            app_passes.append(split_passes >= 2)

        req_passes = []
        for index, atom in enumerate(edge.get("requirements", [])):
            split_passes = 0
            for split_name, frame in splits.items():
                active, _, _ = masks(frame, edge)
                atom_failed = ~predicate_mask(frame, atom)
                others_satisfied = np.ones(len(frame), dtype=bool)
                for other_index, other in enumerate(edge.get("requirements", [])):
                    if other_index != index:
                        others_satisfied &= predicate_mask(frame, other)
                unique_failures = int((active & atom_failed & others_satisfied).sum())
                necessary = unique_failures >= 3
                split_passes += int(necessary)
                atom_rows.append({
                    "candidate_id": edge["candidate_id"], "atom_type": "requirement",
                    "atom_index": index, "atom": atom_text(atom), "split": split_name,
                    "base_support": int(active.sum()), "relaxed_support": np.nan,
                    "new_region_support": np.nan, "new_region_violations": unique_failures,
                    "base_violation_rate": stats(frame, edge)["violation_rate"],
                    "relaxed_violation_rate": np.nan, "rate_increase": np.nan,
                    "necessary_on_split": necessary,
                })
            req_passes.append(split_passes >= 2)
        edge_rows.append({
            "candidate_id": edge["candidate_id"], "description": edge["description"],
            "arity": len({atom.get("left_column") for atom in edge.get("applicability", []) + edge.get("requirements", [])}),
            "applicability_atoms": len(app_passes), "necessary_applicability_atoms": sum(app_passes),
            "requirement_atoms": len(req_passes), "nonredundant_requirement_atoms": sum(req_passes),
            "empirically_minimal": all(app_passes) and all(req_passes),
        })

    atom_frame = pd.DataFrame(atom_rows); edge_frame = pd.DataFrame(edge_rows)
    atom_frame.to_csv(args.output / "atom_leave_one_out.csv", index=False)
    edge_frame.to_csv(args.output / "hyperedge_minimality.csv", index=False)
    lines = [
        "# Hyperedge minimality audit", "",
        "This training-only leave-one-atom-out audit detects inflated conjunctions; it does not establish semantic validity.", "",
        "| Edge | Arity | Necessary antecedents | Nonredundant requirements | Minimal | Description |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in edge_frame.itertuples(index=False):
        lines.append(f"| {row.candidate_id} | {row.arity} | {row.necessary_applicability_atoms}/{row.applicability_atoms} | "
                     f"{row.nonredundant_requirement_atoms}/{row.requirement_atoms} | {bool(row.empirically_minimal)} | {row.description} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "dataset": args.dataset, "seed": args.seed, "selection_uses_heldout": False,
        "real_rows_emitted": False, "edges_audited": len(edge_rows),
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
