#!/usr/bin/env python3
"""Anchor graph-consistent selection to a frozen, train-only C3 subset.

The frozen C3 subset supplies train-only quality and graph-loss anchors. The
selector searches the same particle pool for a subset that reaches a declared
graph-relation improvement while keeping graph coverage/hard excess non-worse
and C3 quality within a declared tolerance. This replaces a scalar graph
penalty with explicit, auditable constraints.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from balanced_particle_resampling import (
    build_geometry_coverage,
    build_groups,
    build_integrated_coverage,
    canonicalize,
    moment_loss,
    rake,
    sha256,
)
from graph_consistent_particle_resampling import (
    consensus_weights,
    graph_groups,
    graph_loss,
    load_graph_factors,
)


def load_indices(path: Path, pool_rows: int, count: int) -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["pool_index"]:
        raise ValueError(f"invalid anchor index schema: {path}")
    indices = frame["pool_index"].to_numpy(np.int64)
    if (len(indices) != count or len(np.unique(indices)) != count
            or indices.min(initial=0) < 0 or indices.max(initial=-1) >= pool_rows):
        raise ValueError(f"invalid anchor indices: {path}")
    return np.sort(indices)


def base_metrics(indices, generic_groups, factors, args, geometry) -> dict:
    generic, univariate, pair, rare = moment_loss(
        indices, generic_groups, args.pair_weight, args.coverage_weight)
    relation, coverage, hard, edge_records = graph_loss(
        indices, factors, args.graph_support_tolerance, args.graph_hard_tolerance)
    geometric_score = geometry.score(indices, args.pool_rows) if geometry else 0.0
    geometric_deficit = (
        max(geometry.reference_score - args.geometric_coverage_tolerance - geometric_score, 0.0)
        if geometry else 0.0)
    quality = generic + args.geometric_coverage_weight * geometric_deficit
    return {
        "quality": float(quality),
        "generic_objective": float(generic),
        "univariate_l1": float(univariate),
        "pair_l1": float(pair),
        "rare_cell_coverage_loss": float(rare),
        "graph_relation_l1": relation,
        "graph_coverage_gap": coverage,
        "graph_hard_excess": hard,
        "geometric_coverage": geometric_score,
        "geometric_coverage_deficit": geometric_deficit,
        "integrated_alpha": None,
        "integrated_coverage": None,
        "integrated_coverage_deficit": None,
        "edge_records": edge_records,
    }


def add_integrated(record: dict, indices: np.ndarray, integrated, args) -> None:
    if integrated is None:
        return
    alpha, beta = integrated.score(indices)
    deficit = max(
        integrated.reference_score - args.integrated_coverage_tolerance - beta, 0.0)
    record["integrated_alpha"] = alpha
    record["integrated_coverage"] = beta
    record["integrated_coverage_deficit"] = deficit
    record["quality"] += args.integrated_coverage_weight * deficit


def constraint_record(record: dict, anchor: dict, args) -> dict:
    relation_target = anchor["graph_relation_l1"] * (1.0 - args.required_graph_gain)
    quality_limit = (
        anchor["quality"] * (1.0 + args.quality_relative_tolerance)
        + args.quality_absolute_tolerance)
    relation_excess = max(record["graph_relation_l1"] - relation_target, 0.0)
    coverage_excess = max(
        record["graph_coverage_gap"] - anchor["graph_coverage_gap"]
        - args.graph_constraint_tolerance, 0.0)
    hard_excess = max(
        record["graph_hard_excess"] - anchor["graph_hard_excess"]
        - args.graph_constraint_tolerance, 0.0)
    quality_excess = max(record["quality"] - quality_limit, 0.0)
    normalized = (
        relation_excess / max(anchor["graph_relation_l1"], 1e-6)
        + coverage_excess / max(anchor["graph_coverage_gap"], 1e-6)
        + hard_excess / max(anchor["graph_hard_excess"], 1e-6)
        + quality_excess / max(anchor["quality"], 1e-6))
    return {
        "relation_target": relation_target,
        "quality_limit": quality_limit,
        "relation_excess": relation_excess,
        "coverage_excess": coverage_excess,
        "hard_excess_vs_anchor": hard_excess,
        "quality_excess": quality_excess,
        "constraint_violation": float(normalized),
        "feasible": normalized <= 1e-12,
    }


def select(weights, anchor_indices, generic_groups, factors, args, geometry, integrated):
    probabilities = np.power(np.clip(weights, 1e-12, None), args.temperature)
    probabilities /= probabilities.sum()
    probabilities = (1.0 - args.tail_mix) * probabilities + args.tail_mix / len(probabilities)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(args.seed)

    anchor = base_metrics(anchor_indices, generic_groups, factors, args, geometry)
    add_integrated(anchor, anchor_indices, integrated, args)
    anchor.update(constraint_record(anchor, anchor, args))

    indices_by_draw: list[np.ndarray] = []
    records: list[dict] = []
    for draw in range(args.draws):
        indices = np.sort(rng.choice(
            len(weights), size=args.count, replace=False, p=probabilities))
        record = base_metrics(indices, generic_groups, factors, args, geometry)
        record["draw"] = draw
        indices_by_draw.append(indices)
        records.append(record)

    # Integrated coverage is expensive. Evaluate the train-only best candidates
    # that already satisfy the graph constraints, with fail-closed fallback.
    relation_target = anchor["graph_relation_l1"] * (1.0 - args.required_graph_gain)
    graph_candidates = [index for index, record in enumerate(records) if (
        record["graph_relation_l1"] <= relation_target + 1e-12
        and record["graph_coverage_gap"] <= anchor["graph_coverage_gap"]
        + args.graph_constraint_tolerance
        and record["graph_hard_excess"] <= anchor["graph_hard_excess"]
        + args.graph_constraint_tolerance)]
    if not graph_candidates:
        graph_candidates = list(range(args.draws))
    evaluated = sorted(
        graph_candidates,
        key=lambda index: (records[index]["quality"],
                           records[index]["graph_relation_l1"], index))
    if integrated is not None:
        evaluated = evaluated[:args.integrated_coverage_shortlist]
        for index in evaluated:
            add_integrated(records[index], indices_by_draw[index], integrated, args)
    for index in evaluated:
        records[index].update(constraint_record(records[index], anchor, args))
    feasible = [index for index in evaluated if records[index]["feasible"]]
    if feasible:
        best = min(feasible, key=lambda index: (
            records[index]["quality"], records[index]["graph_relation_l1"], index))
    else:
        best = min(evaluated, key=lambda index: (
            records[index]["constraint_violation"], records[index]["quality"], index))
    return indices_by_draw[best], {
        "anchor": anchor,
        "best": records[best],
        "feasible_draws": len(feasible),
        "graph_candidate_draws": len(graph_candidates),
        "evaluated_draws": len(evaluated),
        "effective_sample_size": float(1.0 / np.square(probabilities).sum()),
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--backbone-pool", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--anchor-index", type=Path, required=True)
    parser.add_argument("--include-edge-ids", default="")
    parser.add_argument("--include-roles", default="actuate")
    parser.add_argument("--graph-min-support", type=int, default=30)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--pair-bins", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--target-pair-quota", type=int, default=8)
    parser.add_argument("--pair-weight", type=float, default=1.5)
    parser.add_argument("--coverage-weight", type=float, default=.30)
    parser.add_argument("--generic-rake-iterations", type=int, default=5)
    parser.add_argument("--generic-rake-damping", type=float, default=.5)
    parser.add_argument("--graph-rake-iterations", type=int, default=1)
    parser.add_argument("--graph-rake-damping", type=float, default=.10)
    parser.add_argument("--graph-support-tolerance", type=float, default=.005)
    parser.add_argument("--graph-hard-tolerance", type=float, default=.005)
    parser.add_argument("--graph-constraint-tolerance", type=float, default=1e-8)
    parser.add_argument("--required-graph-gain", type=float, required=True)
    parser.add_argument("--quality-relative-tolerance", type=float, required=True)
    parser.add_argument("--quality-absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--prior-index", action="append", type=Path, default=[])
    parser.add_argument("--prior-weight", action="append", type=float, default=[])
    parser.add_argument("--consensus-strength", type=float, default=4.0)
    parser.add_argument("--geometric-coverage-weight", type=float, default=0.0)
    parser.add_argument("--geometric-coverage-k", type=int, default=16)
    parser.add_argument("--geometric-coverage-tolerance", type=float, default=0.0)
    parser.add_argument("--integrated-coverage-weight", type=float, default=0.0)
    parser.add_argument("--integrated-coverage-tolerance", type=float, default=.005)
    parser.add_argument("--integrated-coverage-shortlist", type=int, default=8)
    parser.add_argument("--integrated-coverage-folds", type=int, default=2)
    parser.add_argument("--weight-clip", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=.50)
    parser.add_argument("--tail-mix", type=float, default=.10)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if (not 0 <= args.required_graph_gain < 1
            or args.quality_relative_tolerance < 0
            or args.draws <= 0 or not 0 < args.count):
        raise ValueError("invalid anchored-selection hyperparameters")

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_path = args.repo / "synthetic" / args.dataset / "real.csv"
    train = canonicalize(pd.read_csv(train_path), info)
    pool = canonicalize(pd.read_csv(args.pool), info)
    backbone = canonicalize(pd.read_csv(args.backbone_pool), info)
    if len(pool) != len(backbone) or len(pool) <= args.count:
        raise ValueError("pool/backbone mismatch or no oversampling")
    args.pool_rows = len(pool)
    anchor_indices = load_indices(args.anchor_index, len(pool), args.count)
    edge_ids = {value for value in args.include_edge_ids.split(",") if value}
    roles = {value for value in args.include_roles.split(",") if value}
    factors, skipped = load_graph_factors(
        train, pool, args.graph, edge_ids, roles, args.graph_min_support)
    generic_groups = build_groups(
        train, pool, info, args.bins, args.pair_bins,
        args.max_pairs, args.max_pairs > 0, args.target_pair_quota)
    weights, prior = consensus_weights(
        len(pool), args.count, args.prior_index, args.prior_weight,
        args.consensus_strength, args.weight_clip)
    univariate = [group for group in generic_groups if group.family == "univariate"]
    generic_trace = []
    if args.generic_rake_iterations:
        weights, generic_trace = rake(
            weights, univariate, args.generic_rake_iterations,
            args.generic_rake_damping, args.weight_clip)
    graph_trace = []
    if args.graph_rake_iterations:
        weights, graph_trace = rake(
            weights, graph_groups(factors), args.graph_rake_iterations,
            args.graph_rake_damping, args.weight_clip)
    geometry = (
        build_geometry_coverage(train, pool, info, args.count, args.geometric_coverage_k)
        if args.geometric_coverage_weight > 0 else None)
    integrated = (
        build_integrated_coverage(
            train, pool, info, args.count, args.seed, args.integrated_coverage_folds)
        if args.integrated_coverage_weight > 0 else None)
    selected, selection = select(
        weights, anchor_indices, generic_groups, factors, args, geometry, integrated)

    args.output.mkdir(parents=True)
    pool.iloc[selected].reset_index(drop=True).to_csv(args.output / "selected.csv", index=False)
    pool.iloc[:args.count].reset_index(drop=True).to_csv(args.output / "g_reference.csv", index=False)
    backbone.iloc[:args.count].reset_index(drop=True).to_csv(args.output / "b_reference.csv", index=False)
    pd.DataFrame({"pool_index": selected}).to_csv(
        args.output / "selected_indices.csv", index=False)
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "seed": args.seed,
        "train_rows": len(train), "pool_rows": len(pool), "selected_rows": len(selected),
        "train_sha256": sha256(train_path), "pool_sha256": sha256(args.pool),
        "backbone_pool_sha256": sha256(args.backbone_pool),
        "graph_sha256": sha256(args.graph), "anchor_index_sha256": sha256(args.anchor_index),
        "parameters": {key: value for key, value in vars(args).items()
                       if key not in {"repo", "pool", "backbone_pool", "graph",
                                      "anchor_index", "output", "pool_rows"}},
        "graph_factors": [{
            "candidate_id": factor.candidate_id, "weight": factor.weight,
            "hard": factor.hard, "confidence": factor.confidence,
            "train_support": factor.train_support,
            "train_violation_rate": factor.train_violation_rate,
        } for factor in factors],
        "skipped_graph_factors": skipped,
        "prior": prior, "generic_rake_trace": generic_trace,
        "graph_rake_trace": graph_trace, "selection": selection,
        "geometry_reference": geometry.reference_score if geometry else None,
        "integrated_reference": integrated.reference_score if integrated else None,
        "integrated_fold_seeds": list(integrated.fold_seeds) if integrated else [],
        "weight_summary": {"min": float(weights.min()), "max": float(weights.max()),
                           "mean": float(weights.mean()), "std": float(weights.std())},
    }
    (args.output / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n")
    print(args.output / "selected.csv")
    print(json.dumps({
        "feasible": selection["best"]["feasible"],
        "feasible_draws": selection["feasible_draws"],
        "anchor_quality": selection["anchor"]["quality"],
        "best_quality": selection["best"]["quality"],
        "anchor_graph_relation": selection["anchor"]["graph_relation_l1"],
        "best_graph_relation": selection["best"]["graph_relation_l1"],
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
