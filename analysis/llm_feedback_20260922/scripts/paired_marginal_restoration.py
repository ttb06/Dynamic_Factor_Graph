#!/usr/bin/env python3
"""Restore paired-generator marginals while retaining guided row ordering.

The guided table supplies row ranks (the dependence/copula learned by the
graph action), while the paired unconditional table supplies the exact value
multiset for every acted numerical column.  This is a deterministic optimal
one-dimensional assignment: marginal fidelity is identical to the paired
backbone for acted columns, but graph-induced rank/dependence changes remain.
No held-out row or evaluation metric is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def edge_columns(edge: dict) -> list[str]:
    program = edge.get("program", {})
    kind = program.get("kind")
    if kind in {"ordered_triplet", "sum_equal"}:
        return list(program.get("columns", []))
    if kind == "abs_equal":
        return [program["source"], program["target_column"]]
    if kind == "less_equal":
        return [program["left"], program["right"]]
    columns = list(edge.get("columns", []))
    for atom in edge.get("applicability", []) + edge.get("requirements", []):
        if atom.get("left_column"):
            columns.append(atom["left_column"])
        if atom.get("right_kind") == "column" and atom.get("right_column"):
            columns.append(atom["right_column"])
    return list(dict.fromkeys(columns))


def restore_column(reference: pd.Series, guided: pd.Series) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce").to_numpy(float)
    score = pd.to_numeric(guided, errors="coerce").to_numpy(float)
    if np.isnan(ref).any() or np.isnan(score).any():
        raise ValueError(f"NaN encountered in numerical restoration for {guided.name}")
    # Stable tie-breaking makes the transform deterministic.  Assigning the
    # sorted reference multiset by guided order preserves every reference
    # value and the maximum amount of guided rank information.
    order = np.argsort(score, kind="mergesort")
    restored = np.empty(len(score), dtype=float)
    restored[order] = np.sort(ref, kind="mergesort")
    return pd.Series(restored, index=guided.index, name=guided.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True,
                        help="Paired unconditional sample supplying marginals.")
    parser.add_argument("--guided", type=Path, required=True,
                        help="Guided/repaired sample supplying row ranks.")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--include-families", default="")
    parser.add_argument("--all-numeric", action="store_true",
                        help="Restore every numerical column, not only graph endpoints.")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Convex restoration strength in [0,1].")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must lie in [0,1]")

    reference = pd.read_csv(args.reference)
    guided = pd.read_csv(args.guided)
    if len(reference) != len(guided) or list(reference.columns) != list(guided.columns):
        raise ValueError("reference and guided tables must have identical shape/schema")
    graph = json.loads(args.graph.read_text())
    families = {value.strip() for value in args.include_families.split(",") if value.strip()}
    if families:
        graph = [edge for edge in graph if edge.get("relation_family") in families]
    graph_columns = set(column for edge in graph for column in edge_columns(edge))
    numeric = [column for column in guided.columns
               if pd.api.types.is_numeric_dtype(guided[column])]
    columns = numeric if args.all_numeric else [column for column in numeric if column in graph_columns]
    if not columns:
        raise ValueError("no numerical graph columns selected for restoration")

    result = guided.copy()
    before_l1 = 0.0
    for column in columns:
        old = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        target = restore_column(reference[column], result[column]).to_numpy(float)
        restored = old + args.strength * (target - old)
        before_l1 += float(np.abs(restored - old).sum())
        result[column] = restored
    # Preserve integral storage where the paired backbone uses integers and
    # every restored value is integral.
    for column in columns:
        if pd.api.types.is_integer_dtype(reference[column].dtype):
            result[column] = np.rint(result[column]).astype(reference[column].dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    metadata = {
        "reference": str(args.reference), "guided": str(args.guided),
        "graph": str(args.graph), "families": sorted(families),
        "all_numeric": args.all_numeric, "strength": args.strength, "columns": columns,
        "rows": len(result), "restoration_l1": before_l1,
        "uses_heldout": False, "uses_evaluation_metric": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
