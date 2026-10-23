#!/usr/bin/env python3
"""Apply the same typed rank operator once or repeatedly after generation.

This is the matched post-generation baseline for trajectory rank transport.
It deliberately preserves every acted column's empirical marginal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rank_transport(current: np.ndarray, score: np.ndarray, strength: float) -> np.ndarray:
    n = len(current)
    current_order = np.argsort(current, kind="mergesort")
    score_order = np.argsort(score, kind="mergesort")
    current_rank = np.empty(n, dtype=float); current_rank[current_order] = np.arange(n)
    score_rank = np.empty(n, dtype=float); score_rank[score_order] = np.arange(n)
    blended_rank = (1.0 - strength) * current_rank + strength * score_rank
    assignment_order = np.argsort(blended_rank, kind="mergesort")
    result = np.empty(n, dtype=current.dtype)
    result[assignment_order] = np.sort(current)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--family", default="documented_summary_order")
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.sample)
    original_dtypes = source.dtypes.to_dict()
    result = source.copy()
    edges = [edge for edge in json.loads(args.graph.read_text())
             if edge.get("relation_family") == args.family]
    if not edges:
        raise ValueError(f"no edges found for family {args.family!r}")
    for _ in range(args.passes):
        for edge in edges:
            program = edge.get("program", {})
            if program.get("kind") != "ordered_triplet":
                raise ValueError(f"unsupported final-rank program: {program.get('kind')}")
            left, middle, right = program["columns"]
            left_value = pd.to_numeric(result[left], errors="coerce").fillna(0).to_numpy(float)
            right_value = pd.to_numeric(result[right], errors="coerce").fillna(0).to_numpy(float)
            current = pd.to_numeric(result[middle], errors="coerce").fillna(0).to_numpy(float)
            result[middle] = rank_transport(current, (left_value + right_value) / 2.0,
                                            args.strength)
    # Preserve the source representation as well as the mathematical
    # marginal; this avoids tied-value KS artifacts from float promotion.
    result = result.astype(original_dtypes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    metadata = {
        "sample": str(args.sample), "graph": str(args.graph),
        "family": args.family, "strength": args.strength,
        "passes": args.passes, "rows": len(result), "edges": len(edges),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
