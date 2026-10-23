#!/usr/bin/env python3
"""Convert an LCATS constraint list to a graph and audit its expressiveness."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd


def implication_parts(formula: str) -> tuple[str | None, str]:
    text = str(formula)
    if text.startswith("implies(") and text.endswith(")"):
        inside = text[len("implies("):-1]
        depth = 0
        for index, char in enumerate(inside):
            depth += char in "(["
            depth -= char in ")]"
            if char == "," and depth == 0:
                return inside[:index].strip(), inside[index + 1:].strip()
    return None, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    constraints = json.loads(args.constraints.read_text())
    edges, nodes = [], set()
    for item in constraints:
        columns = list(item.get("cols", []))
        nodes.update(columns)
        antecedent, consequent = implication_parts(item.get("formula", ""))
        edges.append({
            "edge_id": item.get("id"), "description": item.get("desc"),
            "formula": item.get("formula"), "constraint_type": item.get("type"),
            "arity": len(columns), "columns": "|".join(columns),
            "activation_kind": "state_predicate" if antecedent else "always_on",
            "antecedent": antecedent or "", "consequent": consequent,
            "dynamic_ready": bool(antecedent),
        })
    frame = pd.DataFrame(edges)
    frame.to_csv(args.output / "lcats_graph_edges.csv", index=False)
    pd.DataFrame({"node": sorted(nodes)}).to_csv(args.output / "lcats_graph_nodes.csv", index=False)
    counts = frame.constraint_type.value_counts().to_dict()
    lines = [
        "# LCATS discovered-graph audit", "",
        f"- Nodes (columns): **{len(nodes)}**",
        f"- Constraints/edges: **{len(frame)}**",
        f"- Unary constraints: **{int(frame.arity.eq(1).sum())}**",
        f"- Multi-column constraints: **{int(frame.arity.gt(1).sum())}**",
        f"- Explicit state-predicate/conditional edges: **{int(frame.dynamic_ready.sum())}**",
        "", "## Constraint families", "",
        "| Type | Count |", "| --- | ---: |",
    ]
    for name, count in sorted(counts.items()):
        lines.append(f"| {name} | {count} |")
    lines += ["", "## Edges", "", "| ID | Type | Activation | Columns | Description |",
              "| ---: | --- | --- | --- | --- |"]
    for row in frame.itertuples(index=False):
        lines.append(f"| {row.edge_id} | {row.constraint_type} | {row.activation_kind} | {row.columns} | {row.description} |")
    lines += [
        "", "## Implication for the new method", "",
        "LCATS provides a useful propose–execute–validate loop, but the resulting Insurance graph is dominated by always-on unary domain constraints. The reusable extension is to retain nonzero but calibrated relations, learn state predicates, and instantiate different active subgraphs along the diffusion trajectory.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
