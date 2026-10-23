#!/usr/bin/env python3
"""Merge documented functional programs with public News inequalities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    graph = json.loads(args.functional.read_text())
    public = json.loads(args.public.read_text())
    for edge in public:
        atom = edge["requirements"][0]
        graph.append({
            **edge,
            "columns": [atom["left_column"], atom["right_column"]],
            "program": {"kind": "less_equal", "left": atom["left_column"],
                        "right": atom["right_column"],
                        "tolerance": float(atom.get("tolerance", 0.0))},
            "semantic_authority": True,
            "usage_role": "actuate",
        })
    (args.output/"graph.json").write_text(json.dumps(graph, indent=2))
    metadata = {
        "dataset": "news", "functional_programs": len(graph)-len(public),
        "public_inequalities": len(public), "total": len(graph),
        "authority": "independent documented/public specifications",
    }
    (args.output/"run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
