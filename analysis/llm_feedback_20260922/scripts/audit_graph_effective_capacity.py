#!/usr/bin/env python3
"""Audit whether a verified graph has enough distinct, usable actuation signal."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def value_key(value):
    if isinstance(value, bool):
        return ("bool", value)
    try:
        numeric = float(value)
        if math.isfinite(numeric):
            return ("number", round(numeric, 12))
    except (TypeError, ValueError):
        pass
    return ("string", str(value).strip())


def atom_key(atom: dict) -> tuple:
    right = ("column", atom.get("right_column")) if atom.get("right_kind") == "column" else value_key(atom.get("right_value"))
    return atom.get("left_column"), atom.get("operator"), atom.get("right_kind"), right


def program_key(edge: dict, include_exceptions: bool = True) -> tuple:
    key = (
        tuple(sorted(atom_key(x) for x in edge.get("applicability", []))),
        tuple(sorted(atom_key(x) for x in edge.get("requirements", []))),
    )
    if not include_exceptions:
        return key
    exceptions = tuple(sorted(tuple(sorted(atom_key(x) for x in group.get("all_of", [])))
                              for group in edge.get("exceptions", [])))
    return key + (exceptions,)


def endpoint_key(edge: dict) -> tuple:
    left = tuple(sorted({x.get("left_column") for x in edge.get("applicability", [])}))
    right = set()
    for atom in edge.get("requirements", []):
        right.add(atom.get("left_column"))
        if atom.get("right_kind") == "column":
            right.add(atom.get("right_column"))
    return left, tuple(sorted(right))


def duplicate_groups(edges: list[dict], key_fn) -> list[list[str]]:
    groups = {}
    for edge in edges:
        groups.setdefault(key_fn(edge), []).append(edge["candidate_id"])
    return [ids for ids in groups.values() if len(ids) > 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    catalogs = {
        "shoppers": args.outputs/"expanded_constraint_catalog_1057680/constraint_catalog.json",
        "default": args.outputs/"cross_dataset_catalog_default_1058707_1/constraint_catalog.json",
        "diabetes": args.outputs/"diabetes_catalog_1060368/constraint_catalog.json",
    }
    rows, details = [], {}
    for dataset, path in catalogs.items():
        graph = json.loads(path.read_text())
        actuate = [edge for edge in graph if edge.get("usage_role") == "actuate"]
        score = [edge for edge in graph if edge.get("usage_role") == "score_only"]
        info = json.loads((args.repo/"data"/dataset/"info.json").read_text())
        numerical = {info["column_names"][i] for i in info["num_col_idx"]}
        actuation_columns = Counter()
        for edge in actuate:
            for atom in edge.get("requirements", []):
                actuation_columns["numeric" if atom.get("left_column") in numerical else "categorical"] += 1
        exact = duplicate_groups(actuate, lambda edge: program_key(edge, True))
        same_core = duplicate_groups(actuate, lambda edge: program_key(edge, False))
        endpoint = duplicate_groups(actuate, endpoint_key)
        action_counts = Counter()
        traces = list(args.outputs.glob(f"transition_policy_{dataset}_immediate_s*_*/trace_x0_reward_calibrated.json"))
        graph_weights = []
        for trace in traces:
            stats = json.loads(trace.read_text())
            action_counts.update(stats.get("actions_by_edge", {}))
            metadata_path = trace.parent/"run_metadata.json"
            if metadata_path.exists():
                graph_weights.append(float(json.loads(metadata_path.read_text()).get("graph_weight", 0)))
        used = sum(count > 0 for count in action_counts.values())
        total_actions = sum(action_counts.values())
        hhi = (sum((count/total_actions)**2 for count in action_counts.values()) if total_actions else math.nan)
        zero_calibration = sum(float(edge.get("calibration", {}).get("violation_rate", math.nan)) == 0 for edge in actuate)
        row = {
            "dataset": dataset,
            "catalog_edges": len(graph), "actuator_edges": len(actuate), "score_only_edges": len(score),
            "unique_exact_actuators": len({program_key(edge, True) for edge in actuate}),
            "unique_core_actuators": len({program_key(edge, False) for edge in actuate}),
            "unique_endpoint_patterns": len({endpoint_key(edge) for edge in actuate}),
            "zero_calibration_violation_actuators": zero_calibration,
            "numeric_requirement_atoms": actuation_columns["numeric"],
            "categorical_requirement_atoms": actuation_columns["categorical"],
            "edges_used_in_three_immediate_runs": used,
            "total_action_events": total_actions,
            "action_concentration_hhi": hhi,
            "score_guardrail_weight_in_existing_runs": max(graph_weights) if graph_weights else math.nan,
        }
        rows.append(row)
        details[dataset] = {
            "exact_duplicate_groups": exact,
            "same_core_groups": same_core,
            "same_endpoint_groups": endpoint,
            "actions_by_edge": dict(action_counts),
        }
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/"graph_capacity.csv", index=False)
    (args.output/"graph_capacity_details.json").write_text(json.dumps(details, indent=2))
    lines = ["# Effective graph-capacity audit", "", markdown_table(frame), "",
             "Interpretation: exact/core duplicates do not provide independent guidance; HHI near 1 means actions concentrate on very few edges. Existing graph-weight 0 means score-only relations were diagnostic rather than active guardrails."]
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
