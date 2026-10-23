#!/usr/bin/env python3
"""Build controlled graph-size/connectivity variants without held-out selection.

The variants separate three questions that are otherwise easy to confound:

* does merely giving the action selector more guardrails help (role expansion)?
* does adding an edge that joins disconnected schema components help?
* does a denser, connected guardrail graph help beyond either change alone?

New empirical edges are never granted actuation authority.  They can only
penalize candidate actions that introduce a new violation.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from curate_expanded_constraint_catalog import (
    SEMANTIC_ALIASES,
    core_program_key,
    exact_program_key,
    executable,
    finite,
    split_spread,
)


def edge_columns(edge: dict) -> set[str]:
    columns: set[str] = set()
    atoms = list(edge.get("applicability", [])) + list(edge.get("requirements", []))
    atoms += [atom for group in edge.get("exceptions", []) for atom in group.get("all_of", [])]
    for atom in atoms:
        if atom.get("left_column"):
            columns.add(str(atom["left_column"]))
        if atom.get("right_kind") == "column" and atom.get("right_column"):
            columns.add(str(atom["right_column"]))
    return columns


def components(edges: list[dict]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for edge in edges:
        cols = sorted(edge_columns(edge))
        for col in cols:
            find(col)
        for col in cols[1:]:
            union(cols[0], col)
    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return sorted((sorted(value) for value in groups.values()), key=lambda value: (-len(value), value))


def graph_stats(name: str, edges: list[dict]) -> dict:
    nodes = sorted(set().union(*(edge_columns(edge) for edge in edges))) if edges else []
    links = set()
    arities = []
    for edge in edges:
        cols = sorted(edge_columns(edge)); arities.append(len(cols))
        for i, left in enumerate(cols):
            for right in cols[i + 1:]:
                links.add((left, right))
    comps = components(edges)
    possible = len(nodes) * (len(nodes) - 1) / 2
    roles = Counter(edge.get("usage_role", "actuate") for edge in edges)
    return {
        "variant": name,
        "edges": len(edges),
        "nodes": len(nodes),
        "unique_undirected_links": len(links),
        "density": len(links) / possible if possible else 0.0,
        "components": len(comps),
        "largest_component": max((len(value) for value in comps), default=0),
        "actuate": roles["actuate"],
        "score_only": roles["score_only"],
        "monitor": roles["monitor"],
        "multi_column_arity_ge_3": sum(value >= 3 for value in arities),
        "component_members": comps,
    }


def score_copy(edge: dict, rationale: str) -> dict:
    item = copy.deepcopy(edge)
    item["usage_role"] = "score_only"
    item["role_rationale"] = rationale
    item["selection_weight"] = 10.0
    return item


def eligible_extra(edge: dict, min_support: int) -> bool:
    """Discovery/calibration-only gate; heldout is deliberately ignored."""
    if edge.get("source") != "empirical_mining" or not executable(edge):
        return False
    calibration = edge.get("calibration", {})
    return (
        int(calibration.get("support", 0)) >= min_support
        and finite(calibration.get("violation_rate")) <= .22
        and split_spread(edge) <= .06
        and finite(calibration.get("lift"), 0.0) >= 1.10
    )


def unique_extras(source: list[dict], base: list[dict], min_support: int) -> list[dict]:
    exact = {exact_program_key(edge) for edge in base}
    core = {core_program_key(edge) for edge in base}
    base_ids = {edge["candidate_id"] for edge in base}
    selected = []
    for edge in source:
        if edge["candidate_id"] in base_ids or not eligible_extra(edge, min_support):
            continue
        alias = SEMANTIC_ALIASES.get(edge["candidate_id"])
        if alias in base_ids:
            continue
        if exact_program_key(edge) in exact or core_program_key(edge) in core:
            continue
        selected.append(edge)
        exact.add(exact_program_key(edge)); core.add(core_program_key(edge))
    return selected


def joins_components(graph: list[dict], edge: dict) -> bool:
    current = components(graph)
    label = {node: index for index, group in enumerate(current) for node in group}
    touched = edge_columns(edge)
    existing_labels = {label[node] for node in touched if node in label}
    new_nodes = {node for node in touched if node not in label}
    return len(existing_labels) >= 2 or (bool(existing_labels) and bool(new_nodes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--complex-source", type=Path,
                        help="Optional previously reviewed graph containing multi-column hyperedges.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=60)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    source = json.loads(args.source.read_text())
    base = json.loads(args.base.read_text())
    extras = unique_extras(source, base, args.min_support)
    complex_extras = []
    if args.complex_source:
        existing_exact = {exact_program_key(edge) for edge in base}
        existing_core = {core_program_key(edge) for edge in base}
        for raw in json.loads(args.complex_source.read_text()):
            arity = len(edge_columns(raw))
            structurally_complex = (arity >= 3 or len(raw.get("applicability", [])) >= 2
                                    or len(raw.get("requirements", [])) >= 2)
            if (not structurally_complex or not executable(raw)
                    or int(raw.get("calibration", {}).get("support", 0)) < args.min_support
                    or finite(raw.get("judge_confidence"), 0.0) < .65
                    or split_spread(raw) > .06):
                continue
            if exact_program_key(raw) in existing_exact or core_program_key(raw) in existing_core:
                continue
            edge = copy.deepcopy(raw)
            edge["candidate_id"] = f"prior_complex_{raw['candidate_id']}"
            edge["source"] = "prior_reviewed_complex_graph"
            complex_extras.append(edge)
            existing_exact.add(exact_program_key(raw)); existing_core.add(core_program_key(raw))

    # Same edge set as the base graph, but every calibrated diagnostic edge is
    # exposed to the branch selector as a non-actuating guardrail.
    role_expanded = [
        score_copy(edge, "soft diagnostic promoted to non-actuating action guardrail")
        if edge.get("usage_role") == "monitor" else copy.deepcopy(edge)
        for edge in base
    ]

    # Greedily select only edges that reduce the component count.  Sorting is
    # based on calibration support/stability, never held-out performance.
    bridge = copy.deepcopy(base); bridge_ids = []
    ordered = sorted(extras, key=lambda edge: (
        split_spread(edge),
        finite(edge.get("calibration", {}).get("violation_rate")),
        -int(edge.get("calibration", {}).get("support", 0)),
    ))
    for edge in ordered:
        if joins_components(bridge, edge):
            bridge.append(score_copy(edge, "empirical bridge used only as a non-actuating guardrail"))
            bridge_ids.append(edge["candidate_id"])

    # Maximal discovery-safe graph: role expansion plus all stable extra
    # empirical edges.  These edges can veto collateral damage but cannot
    # initiate edits, so density is not mistaken for semantic authority.
    dense = copy.deepcopy(role_expanded)
    dense.extend(score_copy(edge, "stable empirical relation used only as a non-actuating guardrail")
                 for edge in extras)

    validated_hyperedges = copy.deepcopy(base)
    validated_hyperedges.extend(
        score_copy(edge, "previously reviewed multi-column hyperedge used only as an action guardrail")
        for edge in complex_extras
    )
    dense_hypergraph = copy.deepcopy(dense)
    dense_hypergraph.extend(
        score_copy(edge, "previously reviewed multi-column hyperedge used only as an action guardrail")
        for edge in complex_extras
    )

    variants = {
        "base": copy.deepcopy(base),
        "role_expanded": role_expanded,
        "bridge_guardrails": bridge,
        "dense_connected": dense,
        "validated_hyperedges": validated_hyperedges,
        "dense_hypergraph": dense_hypergraph,
    }
    summaries = []
    for name, graph in variants.items():
        (args.output / f"{name}.json").write_text(json.dumps(graph, indent=2))
        summaries.append(graph_stats(name, graph))

    flat = [{key: value for key, value in row.items() if key != "component_members"} for row in summaries]
    pd.DataFrame(flat).to_csv(args.output / "graph_summary.csv", index=False)
    audit = {
        "selection_uses_heldout": False,
        "eligible_extra_ids": [edge["candidate_id"] for edge in extras],
        "reviewed_complex_ids": [edge["candidate_id"] for edge in complex_extras],
        "bridge_ids": bridge_ids,
        "variants": summaries,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    lines = [
        "# Controlled graph size/connectivity variants", "",
        "New empirical edges are score-only guardrails and cannot actuate a repair. Held-out data were not used for selection.", "",
        "| Variant | Edges | Nodes | Links | Density | Components | Largest component | Actuate | Score-only | Monitor |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['edges']} | {row['nodes']} | {row['unique_undirected_links']} | "
            f"{row['density']:.3f} | {row['components']} | {row['largest_component']} | "
            f"{row['actuate']} | {row['score_only']} | {row['monitor']} |"
        )
    lines += ["", f"Eligible extra edges: `{audit['eligible_extra_ids']}`.",
              f"Greedy component-joining edges: `{bridge_ids}`."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
