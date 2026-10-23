#!/usr/bin/env python3
"""Select train-exact, supported relations without consulting held-out data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from external_knowledge_reconstruction_bench import exact_edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--include-score-only", action="store_true")
    args = parser.parse_args()
    train = pd.read_csv(args.repo / "synthetic" / args.dataset / "real.csv")
    graph = json.loads(args.graph.read_text())
    selected = exact_edges(graph, train, args.include_score_only, args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2))
    print(json.dumps({
        "dataset": args.dataset,
        "input_edges": len(graph),
        "selected_exact_edges": len(selected),
        "include_score_only": args.include_score_only,
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
