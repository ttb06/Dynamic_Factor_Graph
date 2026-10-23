#!/usr/bin/env python3
"""Audit fixed output subsets against the dynamic sampler's active graph factors."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from balanced_particle_resampling import canonicalize, sha256
from graph_consistent_particle_resampling import graph_loss, load_graph_factors


def parse_sample(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("sample must be NAME=INDEX_CSV")
    name, filename = value.split("=", 1)
    if not name or not filename:
        raise argparse.ArgumentTypeError("sample must be NAME=INDEX_CSV")
    return name, Path(filename)


def load_indices(path: Path, pool_rows: int, count: int | None) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["pool_index"]:
        raise ValueError(f"invalid index schema: {path}")
    indices = frame["pool_index"].to_numpy(np.int64)
    if (not len(indices) or len(np.unique(indices)) != len(indices)
            or indices.min(initial=0) < 0 or indices.max(initial=-1) >= pool_rows):
        raise ValueError(f"invalid pool indices: {path}")
    if count is not None and len(indices) != count:
        raise ValueError(f"expected {count} indices in {path}, got {len(indices)}")
    return indices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--include-edge-ids", default="")
    parser.add_argument("--include-roles", default="actuate")
    parser.add_argument("--graph-min-support", type=int, default=30)
    parser.add_argument("--graph-support-tolerance", type=float, default=0.005)
    parser.add_argument("--graph-hard-tolerance", type=float, default=0.005)
    parser.add_argument("--count", type=int)
    parser.add_argument("--sample-index", action="append", type=parse_sample, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_path = args.repo / "synthetic" / args.dataset / "real.csv"
    train = canonicalize(pd.read_csv(train_path), info)
    pool = canonicalize(pd.read_csv(args.pool), info)
    edge_ids = {value for value in args.include_edge_ids.split(",") if value}
    roles = {value for value in args.include_roles.split(",") if value}
    factors, skipped = load_graph_factors(
        train, pool, args.graph, edge_ids, roles, args.graph_min_support)

    rows: list[dict] = []
    edge_rows: list[dict] = []
    inputs: list[dict] = []
    for name, path in args.sample_index:
        indices = load_indices(path, len(pool), args.count)
        relation, coverage, hard, edge_records = graph_loss(
            indices, factors, args.graph_support_tolerance,
            args.graph_hard_tolerance)
        rows.append({
            "method": name,
            "rows": len(indices),
            "graph_relation_l1": relation,
            "graph_coverage_gap": coverage,
            "graph_hard_excess": hard,
            "graph_hard_admissible": hard <= 1e-12,
        })
        for record in edge_records:
            edge_rows.append({"method": name, **record})
        inputs.append({"method": name, "path": str(path), "sha256": sha256(path)})

    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output / "graph_metrics.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(args.output / "per_edge_metrics.csv", index=False)
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "train_sha256": sha256(train_path),
        "pool_sha256": sha256(args.pool),
        "graph_sha256": sha256(args.graph),
        "included_edge_ids": sorted(edge_ids),
        "included_roles": sorted(roles),
        "support_tolerance": args.graph_support_tolerance,
        "hard_tolerance": args.graph_hard_tolerance,
        "factors": [{
            "candidate_id": factor.candidate_id,
            "weight": factor.weight,
            "hard": factor.hard,
            "confidence": factor.confidence,
            "train_support": factor.train_support,
            "train_violation_rate": factor.train_violation_rate,
        } for factor in factors],
        "skipped_factors": skipped,
        "inputs": inputs,
    }
    (args.output / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
