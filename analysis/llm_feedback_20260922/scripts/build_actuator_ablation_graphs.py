#!/usr/bin/env python3
"""Create one-actuator graph variants while retaining all graph guardrails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    graph = json.loads(args.graph.read_text())
    actuators = [edge for edge in graph if edge.get("usage_role", "actuate") == "actuate"]
    guardrails = [edge for edge in graph if edge.get("usage_role", "actuate") != "actuate"]
    rows = []
    for index, edge in enumerate(actuators):
        path = args.output / f"actuator_{index:02d}.json"
        path.write_text(json.dumps([edge, *guardrails], indent=2))
        rows.append({
            "index": index,
            "candidate_id": edge["candidate_id"],
            "classification": edge["classification"],
            "relation_family": edge.get("relation_family"),
            "cutoff_graph": str(path),
        })
    pd.DataFrame(rows).to_csv(args.output / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
