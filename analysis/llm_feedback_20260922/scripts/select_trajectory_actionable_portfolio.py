#!/usr/bin/env python3
"""Select a non-redundant actuator portfolio using a calibration trajectory.

The selector never grants new actuation authority.  It only retains a subset
of semantically authorized actuators that are readable, state-changing, and
exposed to generator errors; all held-out fields are ignored.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from curate_expanded_constraint_catalog import core_program_key


def endpoint_key(edge: dict) -> tuple:
    left = tuple(sorted({atom.get("left_column") for atom in edge.get("applicability", [])}))
    right = set()
    for atom in edge.get("requirements", []):
        right.add(atom.get("left_column"))
        if atom.get("right_kind") == "column":
            right.add(atom.get("right_column"))
    return left, tuple(sorted(right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-actuators", type=int, default=8)
    parser.add_argument("--min-final-violations", type=int, default=2)
    parser.add_argument("--max-per-endpoint", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    graph = json.loads(args.graph.read_text())
    trajectory = pd.read_csv(args.trajectory)
    schedule = pd.read_csv(args.schedule).set_index("candidate_id")
    middle = trajectory[(trajectory.t >= .25) & (trajectory.t <= .75)]
    mid_flip = middle.groupby("candidate_id").active_flip.mean().fillna(0).to_dict()
    candidates, decisions = [], []
    for edge in graph:
        edge_id = edge["candidate_id"]
        if edge.get("usage_role") != "actuate":
            continue
        if edge_id not in schedule.index:
            decisions.append({"candidate_id": edge_id, "decision": "demote", "reason": "missing_schedule"})
            continue
        row = schedule.loc[edge_id]
        cutoff = float(row.cutoff) if pd.notna(row.cutoff) else math.nan
        violations = int(row.final_violations)
        readable = bool(row.readable) and math.isfinite(cutoff)
        flip = float(mid_flip.get(edge_id, 0.0))
        # Challenge exposure dominates; middle readability and topology
        # evolution break ties.  This is calibrated on a separate generator
        # seed and never uses held-out real-data statistics.
        score = math.log1p(max(violations, 0)) + 20.0*flip
        if .25 <= cutoff <= .50:
            score += 1.0
        eligible = readable and violations >= args.min_final_violations
        candidates.append((eligible, score, violations, flip, cutoff, edge))

    candidates.sort(key=lambda item: (-int(item[0]), -item[1], -item[2], item[5]["candidate_id"]))
    selected_ids, endpoint_counts, core_seen = set(), {}, set()
    for eligible, score, violations, flip, cutoff, edge in candidates:
        edge_id = edge["candidate_id"]
        endpoint, core = endpoint_key(edge), core_program_key(edge)
        reason = None
        if not eligible:
            reason = "unreadable_or_no_generator_violation_exposure"
        elif core in core_seen:
            reason = "semantic_core_duplicate"
        elif endpoint_counts.get(endpoint, 0) >= args.max_per_endpoint:
            reason = "endpoint_diversity_cap"
        elif len(selected_ids) >= args.max_actuators:
            reason = "actuator_budget"
        if reason:
            decisions.append({"candidate_id": edge_id, "decision": "demote", "reason": reason,
                              "portfolio_score": score, "final_violations": violations,
                              "middle_flip": flip, "cutoff": cutoff})
            continue
        selected_ids.add(edge_id); core_seen.add(core); endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0)+1
        decisions.append({"candidate_id": edge_id, "decision": "retain_actuator", "reason": "trajectory_actionable",
                          "portfolio_score": score, "final_violations": violations,
                          "middle_flip": flip, "cutoff": cutoff})

    portfolio = []
    for edge in graph:
        item = dict(edge)
        if item.get("usage_role") == "actuate" and item["candidate_id"] not in selected_ids:
            item["usage_role"] = "score_only"
            item["role_rationale"] = "semantically valid but not selected by calibration-trajectory actionability portfolio"
            item["selection_weight"] = 10.0
        portfolio.append(item)
    summary = {
        "input_edges": len(graph),
        "input_actuators": sum(edge.get("usage_role") == "actuate" for edge in graph),
        "selected_actuators": len(selected_ids),
        "selected_ids": sorted(selected_ids),
        "uses_heldout": False,
        "selection_data": "separate unconditional calibration trajectory",
    }
    (args.output/"constraint_catalog.json").write_text(json.dumps(portfolio, indent=2))
    (args.output/"portfolio_decisions.json").write_text(json.dumps(decisions, indent=2))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Trajectory-actionable graph portfolio", "", f"Summary: `{summary}`", "",
             "This stage cannot promote a statistical edge to actuator. It only filters previously verified semantic actuators using a calibration trajectory."]
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
