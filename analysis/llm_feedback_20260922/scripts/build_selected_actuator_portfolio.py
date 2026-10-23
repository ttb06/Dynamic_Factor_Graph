#!/usr/bin/env python3
"""Build a middle-guidance portfolio while retaining non-actuating guardrails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--selected", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = json.loads(args.graph.read_text())
    requested = set(args.selected)
    selected = [
        edge for edge in graph
        if edge.get("usage_role", "actuate") == "actuate"
        and edge["candidate_id"] in requested
    ]
    found = {edge["candidate_id"] for edge in selected}
    if found != requested:
        raise ValueError(f"Missing selected actuators: {sorted(requested - found)}")
    guardrails = [edge for edge in graph if edge.get("usage_role", "actuate") != "actuate"]
    portfolio = [*selected, *guardrails]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(portfolio, indent=2))
    metadata = {
        "source_graph": str(args.graph),
        "selected_actuator_ids": args.selected,
        "selected_actuators": len(selected),
        "retained_guardrails": len(guardrails),
        "total_edges": len(portfolio),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
