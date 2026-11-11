#!/usr/bin/env python3
"""Train-only local-manifold membership repair under a frozen C3 quality guard.

The repair never edits a generated row and never reads held-out/test data. It
greedily adds pool rows that cover currently uncovered train neighbourhoods,
then removes nearby selected rows so low-order C3 statistics remain stable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from dygraft import anchoring as bpr
from dygraft.coverage_index import search


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def encoded_pool(train: pd.DataFrame, pool: pd.DataFrame, info: dict) -> np.ndarray:
    numeric, categorical = bpr.typed_columns(info)
    train_numeric = train[numeric].to_numpy(float)
    pool_numeric = pool[numeric].to_numpy(float)
    low = np.nanmin(train_numeric, axis=0)
    span = np.nanmax(train_numeric, axis=0) - low
    span[span == 0] = 1.0
    pool_numeric = (pool_numeric - low) / span
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(train[categorical].astype(str))
        pool_categorical = encoder.transform(pool[categorical].astype(str))
        matrix = np.concatenate([pool_numeric, pool_categorical], axis=1)
    else:
        matrix = pool_numeric
    return np.asarray(matrix, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--backbone-pool", type=Path, required=True)
    parser.add_argument("--anchor-index", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--swap-fraction", type=float, required=True)
    parser.add_argument("--quality-relative-tolerance", type=float, required=True)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--addition-shortlist", type=int, default=8)
    parser.add_argument("--removal-shortlist", type=int, default=16)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--pair-bins", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--target-pair-quota", type=int, default=8)
    parser.add_argument("--pair-weight", type=float, default=1.5)
    parser.add_argument("--coverage-weight", type=float, default=0.30)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (not 0 < args.swap_fraction <= 0.10 or
            args.quality_relative_tolerance < 0 or args.neighbors < 1 or
            args.addition_shortlist < 1 or args.removal_shortlist < 1):
        raise ValueError("invalid repair hyperparameters")

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_path = args.repo / "synthetic" / args.dataset / "real.csv"
    train = bpr.canonicalize(pd.read_csv(train_path), info)
    pool = bpr.canonicalize(pd.read_csv(args.pool), info)
    backbone = bpr.canonicalize(pd.read_csv(args.backbone_pool), info)
    anchor_frame = pd.read_csv(args.anchor_index)
    if list(anchor_frame.columns) != ["pool_index"]:
        raise ValueError("invalid anchor index schema")
    anchor = np.sort(anchor_frame["pool_index"].to_numpy(np.int64))
    if (len(anchor) != args.count or len(np.unique(anchor)) != args.count or
            anchor.min(initial=0) < 0 or anchor.max(initial=-1) >= len(pool) or
            len(pool) != len(backbone)):
        raise ValueError("invalid anchor or pool")

    groups = bpr.build_groups(
        train, pool, info, args.bins, args.pair_bins,
        args.max_pairs, True, args.target_pair_quota)
    anchor_loss = bpr.moment_loss(
        anchor, groups, args.pair_weight, args.coverage_weight)
    quality_limit = anchor_loss[0] * (1.0 + args.quality_relative_tolerance)
    geometry = bpr.build_geometry_coverage(
        train, pool, info, args.count, args.neighbors)
    matrix = encoded_pool(train, pool, info)

    selected = np.zeros(len(pool), dtype=bool)
    selected[anchor] = True
    anchor_geometry = geometry.score(anchor, len(pool))
    pool_ceiling = geometry.score(np.arange(len(pool), dtype=np.int64), len(pool))
    target_swaps = max(1, int(round(args.count * args.swap_fraction)))
    diagnostics = {}
    trace = search(args, geometry, matrix, groups, selected, anchor, quality_limit, target_swaps, diagnostics)

    final_indices = np.flatnonzero(selected)
    if len(final_indices) != args.count:
        raise RuntimeError("repair changed output cardinality")
    final_loss = bpr.moment_loss(
        final_indices, groups, args.pair_weight, args.coverage_weight)
    final_geometry = geometry.score(final_indices, len(pool))
    args.output.mkdir(parents=True)
    pool.iloc[final_indices].reset_index(drop=True).to_csv(
        args.output / "selected.csv", index=False)
    pd.DataFrame({"pool_index": final_indices}).to_csv(
        args.output / "selected_indices.csv", index=False)
    pool.iloc[anchor].reset_index(drop=True).to_csv(
        args.output / "anchor.csv", index=False)
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "seed": args.seed,
        "pool_rows": len(pool), "selected_rows": len(final_indices),
        "target_swaps": target_swaps, "achieved_swaps": len(trace),
        "search_diagnostics": diagnostics,
        "anchor_objective": anchor_loss[0], "final_objective": final_loss[0],
        "quality_limit": quality_limit,
        "anchor_geometry_coverage": anchor_geometry,
        "final_geometry_coverage": final_geometry,
        "pool_geometry_ceiling": pool_ceiling,
        "parameters": vars(args) | {"repo": str(args.repo), "pool": str(args.pool),
                                      "backbone_pool": str(args.backbone_pool),
                                      "anchor_index": str(args.anchor_index),
                                      "output": str(args.output)},
        "hashes": {"pool": sha256(args.pool), "anchor": sha256(args.anchor_index),
                   "train": sha256(train_path)},
        "trace": trace,
    }
    (args.output / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_default) + "\n")
    print(json.dumps({
        "output": str(args.output), "achieved_swaps": len(trace),
        "anchor_geometry": anchor_geometry, "final_geometry": final_geometry,
        "pool_ceiling": pool_ceiling, "anchor_objective": anchor_loss[0],
        "final_objective": final_loss[0]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
