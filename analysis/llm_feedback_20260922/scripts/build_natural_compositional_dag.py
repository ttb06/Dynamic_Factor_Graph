#!/usr/bin/env python3
"""Compile train-selected compositional motifs from a verified typed graph.

The compiler follows exact literal unification: the consequent written by one
edge must equal the antecedent read by the next edge.  It never uses held-out
statistics for path selection.  Roots are supplied by the experiment config;
all descendants are selected from discovery/counterexample/calibration evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TRAIN_SPLITS = ("discovery", "counterexample", "calibration")


def literal_key(atom: dict) -> tuple[str, str, str] | None:
    if atom.get("right_kind") != "literal":
        return None
    value = atom.get("right_value")
    try:
        number = float(value)
        value_key = str(int(number)) if number.is_integer() else f"{number:.12g}"
    except (TypeError, ValueError):
        value_key = str(value)
    return str(atom["left_column"]), str(atom["operator"]), value_key


def train_rates(edge: dict) -> list[float]:
    return [float(edge[name]["violation_rate"]) for name in TRAIN_SPLITS]


def train_score(edge: dict) -> float:
    rates = train_rates(edge)
    support = min(int(edge[name]["support"]) for name in TRAIN_SPLITS)
    lift = min(float(edge[name].get("lift", 1.0)) for name in TRAIN_SPLITS)
    stability = max(rates) - min(rates)
    return math.log1p(support) * min(lift, 8.0) * (1.0 - max(rates)) / (1.0 + 10.0 * stability)


def eligible(edge: dict, args: argparse.Namespace) -> bool:
    if args.source and edge.get("source") != args.source:
        return False
    if len(edge.get("applicability", [])) != 1 or len(edge.get("requirements", [])) != 1:
        return False
    if edge.get("exceptions"):
        return False
    if literal_key(edge["applicability"][0]) is None or literal_key(edge["requirements"][0]) is None:
        return False
    rates = train_rates(edge)
    if max(rates) > args.max_train_violation:
        return False
    if max(rates) - min(rates) > args.max_train_shift:
        return False
    if min(int(edge[name]["support"]) for name in TRAIN_SPLITS) < args.min_support:
        return False
    return True


def creates_cycle(selected: list[dict], candidate: dict) -> bool:
    """Return whether adding a single-literal edge closes a column cycle."""
    source = candidate["applicability"][0]["left_column"]
    target = candidate["requirements"][0]["left_column"]
    adjacency: dict[str, set[str]] = {}
    for edge in selected:
        left = edge["applicability"][0]["left_column"]
        right = edge["requirements"][0]["left_column"]
        adjacency.setdefault(left, set()).add(right)
    frontier = [target]
    visited = set()
    while frontier:
        node = frontier.pop()
        if node == source:
            return True
        if node in visited:
            continue
        visited.add(node)
        frontier.extend(adjacency.get(node, ()))
    return False


def compile_path(root: dict, candidates: list[dict], max_depth: int,
                 graph_selected: list[dict]) -> list[dict]:
    selected_ids = {edge["candidate_id"] for edge in graph_selected}
    if root["candidate_id"] not in selected_ids:
        if creates_cycle(graph_selected, root):
            raise ValueError(f"root {root['candidate_id']} would create a cycle")
        graph_selected.append(root)
        selected_ids.add(root["candidate_id"])
    path = [root]
    touched = {root["applicability"][0]["left_column"], root["requirements"][0]["left_column"]}
    while len(path) < max_depth:
        bridge = literal_key(path[-1]["requirements"][0])
        options = []
        for edge in candidates:
            if edge["candidate_id"] in {item["candidate_id"] for item in path}:
                continue
            if literal_key(edge["applicability"][0]) != bridge:
                continue
            target = edge["requirements"][0]["left_column"]
            if target in touched:
                continue
            if edge["candidate_id"] not in selected_ids and creates_cycle(graph_selected, edge):
                continue
            options.append(edge)
        if not options:
            break
        chosen = max(options, key=lambda edge: (train_score(edge), edge["candidate_id"]))
        path.append(chosen)
        if chosen["candidate_id"] not in selected_ids:
            # Keep the union acyclic even when several root paths merge.
            graph_selected.append(chosen)
            selected_ids.add(chosen["candidate_id"])
        touched.add(chosen["requirements"][0]["left_column"])
    return path


def topological_edges(edges: list[dict]) -> list[dict]:
    """Order parents before children so final-repair controls are not handicapped."""
    by_id = {edge["candidate_id"]: edge for edge in edges}
    parents = {candidate_id: set() for candidate_id in by_id}
    children = {candidate_id: set() for candidate_id in by_id}
    for left in edges:
        produced = literal_key(left["requirements"][0])
        for right in edges:
            if left["candidate_id"] == right["candidate_id"]:
                continue
            if produced == literal_key(right["applicability"][0]):
                parents[right["candidate_id"]].add(left["candidate_id"])
                children[left["candidate_id"]].add(right["candidate_id"])
    ready = sorted((candidate_id for candidate_id, incoming in parents.items() if not incoming),
                   key=lambda candidate_id: (-train_score(by_id[candidate_id]), candidate_id))
    ordered = []
    while ready:
        candidate_id = ready.pop(0)
        ordered.append(by_id[candidate_id])
        for child in sorted(children[candidate_id]):
            parents[child].discard(candidate_id)
            if not parents[child] and child not in {edge["candidate_id"] for edge in ordered} and child not in ready:
                ready.append(child)
        ready.sort(key=lambda item: (-train_score(by_id[item]), item))
    if len(ordered) != len(edges):
        raise ValueError("compiled graph is cyclic under exact literal unification")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-graph", type=Path, required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--source", default="empirical_mining")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--max-train-violation", type=float, default=.25)
    parser.add_argument("--max-train-shift", type=float, default=.05)
    parser.add_argument("--cutoff", type=float, default=.50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    source = json.loads(args.source_graph.read_text())
    by_id = {edge["candidate_id"]: edge for edge in source}
    missing = sorted(set(args.roots) - set(by_id))
    if missing:
        raise ValueError(f"unknown roots: {missing}")
    candidates = [edge for edge in source if eligible(edge, args)]
    paths = []
    graph_selected: list[dict] = []
    for root_id in args.roots:
        root = by_id[root_id]
        if not eligible(root, args):
            raise ValueError(f"root {root_id} fails train-only eligibility filters")
        paths.append(compile_path(root, candidates, args.max_depth, graph_selected))

    graph = []
    for edge in topological_edges(graph_selected):
        item = dict(edge)
        item["usage_role"] = "actuate"
        item["motif_train_score"] = train_score(edge)
        graph.append(item)

    (args.output / "graph.json").write_text(json.dumps(graph, indent=2))
    with (args.output / "schedule.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "candidate_id", "cutoff", "readable", "violation_evaluable"
        ])
        writer.writeheader()
        for edge in graph:
            writer.writerow({"candidate_id": edge["candidate_id"], "cutoff": args.cutoff,
                             "readable": True, "violation_evaluable": True})

    path_ids = [[edge["candidate_id"] for edge in path] for path in paths]
    summary = {
        "status": "train-selected natural compositional DAG",
        "source_graph": str(args.source_graph),
        "roots": args.roots,
        "paths": path_ids,
        "edges": len(graph),
        "selection_uses_heldout": False,
        "selection_evidence": list(TRAIN_SPLITS),
        "unification": "exact consequent literal to antecedent literal",
        "filters": {
            "source": args.source,
            "min_support_each_train_partition": args.min_support,
            "max_train_violation": args.max_train_violation,
            "max_train_shift": args.max_train_shift,
            "exceptions_allowed": False,
        },
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Natural compositional DAG", "", f"Roots: `{args.roots}`", "",
             "Selection used discovery/counterexample/calibration metadata only.", "",
             "| Path | Edge sequence |", "| ---: | --- |"]
    for index, path in enumerate(path_ids, 1):
        lines.append(f"| {index} | {' -> '.join(path)} |")
    lines += ["", "| Edge | Train score | Calibration rate | Held-out rate (audit only) |",
              "| --- | ---: | ---: | ---: |"]
    for edge in graph:
        lines.append(
            f"| {edge['candidate_id']} | {edge['motif_train_score']:.4f} | "
            f"{float(edge['calibration']['violation_rate']):.4f} | "
            f"{float(edge['heldout']['violation_rate']):.4f} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
