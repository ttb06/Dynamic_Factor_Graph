#!/usr/bin/env python3
"""Repair a frozen C3 subset with sparse, factor-directed membership swaps.

The active factor graph defines which selected rows are over-represented and
which unselected rows can close graph-state deficits. Candidate subsets differ
from the frozen C3 anchor by at most a declared swap fraction. Selection is
train-only and fail-closed on generic C3 quality, graph coverage, hard-factor
excess, and a required graph-relation gain.
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
    sha256,
)
from graph_consistent_particle_resampling import (
    graph_loss,
    load_graph_factors,
    smoothed_distribution,
)
from graph_pareto_anchored_resampling import (
    add_integrated,
    base_metrics,
    constraint_record,
    load_indices,
)


def directed_scores(anchor_indices: np.ndarray, factors, pool_rows: int,
                    support_tolerance: float, hard_tolerance: float,
                    coverage_boost: float, hard_boost: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """First-order L1 gains for removing/adding a row under each factor."""
    add = np.zeros(pool_rows, dtype=float)
    remove = np.zeros(pool_rows, dtype=float)
    records: list[dict] = []
    for factor in factors:
        target = smoothed_distribution(factor.train_states)
        observed = smoothed_distribution(factor.pool_states[anchor_indices])
        residual_sign = np.sign(observed - target)
        # Removing an over-represented state and adding an under-represented
        # state are the steepest-descent directions for factor-state L1.
        remove_component = residual_sign[factor.pool_states]
        add_component = -residual_sign[factor.pool_states]

        anchor_states = factor.pool_states[anchor_indices]
        support = float(np.mean(anchor_states > 0))
        support_deficit = max(factor.train_support - support_tolerance - support, 0.0)
        if support_deficit > 0:
            add_component = add_component + coverage_boost * (factor.pool_states > 0)
            remove_component = remove_component + coverage_boost * (factor.pool_states == 0)

        active = anchor_states > 0
        violation = anchor_states == 2
        violation_rate = float(violation.sum() / max(active.sum(), 1))
        hard_excess = (
            max(violation_rate - factor.train_violation_rate - hard_tolerance, 0.0)
            if factor.hard else 0.0)
        if hard_excess > 0:
            remove_component = remove_component + hard_boost * (factor.pool_states == 2)
            add_component = add_component + hard_boost * (factor.pool_states != 2)

        add += factor.weight * add_component
        remove += factor.weight * remove_component
        records.append({
            "candidate_id": factor.candidate_id,
            "weight": factor.weight,
            "target": target.tolist(),
            "anchor": observed.tolist(),
            "support_deficit": support_deficit,
            "hard_excess": hard_excess,
        })
    return add, remove, records


def probabilities(scores: np.ndarray, temperature: float, uniform_mix: float) -> np.ndarray:
    centered = scores - float(np.median(scores))
    scale = float(np.std(centered))
    if scale > 1e-12:
        centered /= scale
    logits = np.clip(centered / temperature, -12.0, 12.0)
    result = np.exp(logits - float(logits.max()))
    result /= result.sum()
    result = (1.0 - uniform_mix) * result + uniform_mix / len(result)
    return result / result.sum()


def repair(anchor_indices, factors, generic_groups, args, geometry, integrated):
    pool_rows = args.pool_rows
    anchor = base_metrics(anchor_indices, generic_groups, factors, args, geometry)
    add_integrated(anchor, anchor_indices, integrated, args)
    if (args.graph_activation_threshold > 0
            and anchor["graph_relation_l1"] <= args.graph_activation_threshold):
        # A relative-gain requirement is ill-conditioned once the frozen C3
        # anchor is already graph-consistent. In that regime the graph acts as
        # a no-op gate: preserve the quality-optimal anchor instead of forcing
        # harmful swaps merely to manufacture a relative improvement.
        best = dict(anchor)
        best.update({
            "draw": -1,
            "swapped_rows": 0,
            "changed_membership": 0,
            "relation_target": anchor["graph_relation_l1"],
            "quality_limit": anchor["quality"],
            "relation_excess": 0.0,
            "coverage_excess": 0.0,
            "hard_excess_vs_anchor": 0.0,
            "quality_excess": 0.0,
            "constraint_violation": 0.0,
            "feasible": True,
            "activation_skipped": True,
        })
        return np.sort(anchor_indices.copy()), {
            "anchor": dict(anchor),
            "best": best,
            "feasible_draws": 1,
            "evaluated_draws": 1,
            "swap_count": 0,
            "swap_fraction_realized": 0.0,
            "activation_skipped": True,
            "activation_reason": "anchor_graph_relation_below_threshold",
            "direction_records": [],
            "add_score_summary": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
            "remove_score_summary": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
            "records": [best],
        }

    selected_mask = np.zeros(pool_rows, dtype=bool)
    selected_mask[anchor_indices] = True
    outside = np.flatnonzero(~selected_mask)
    swap_count = max(1, min(
        int(round(args.count * args.swap_fraction)),
        len(anchor_indices), len(outside)))
    add_score, remove_score, direction_records = directed_scores(
        anchor_indices, factors, pool_rows, args.graph_support_tolerance,
        args.graph_hard_tolerance, args.coverage_boost, args.hard_boost)
    add_prob = probabilities(add_score[outside], args.swap_temperature, args.uniform_mix)
    remove_prob = probabilities(
        remove_score[anchor_indices], args.swap_temperature, args.uniform_mix)
    rng = np.random.default_rng(args.seed)

    anchor.update(constraint_record(anchor, anchor, args))
    candidates: list[np.ndarray] = []
    records: list[dict] = []
    for draw in range(args.draws):
        if draw == 0:
            removed = anchor_indices[np.argsort(remove_score[anchor_indices])[-swap_count:]]
            added = outside[np.argsort(add_score[outside])[-swap_count:]]
        else:
            removed = rng.choice(anchor_indices, size=swap_count, replace=False, p=remove_prob)
            added = rng.choice(outside, size=swap_count, replace=False, p=add_prob)
        indices = np.sort(np.concatenate((
            anchor_indices[~np.isin(anchor_indices, removed, assume_unique=True)], added)))
        record = base_metrics(indices, generic_groups, factors, args, geometry)
        record.update({"draw": draw, "swapped_rows": swap_count,
                       "changed_membership": 2 * swap_count})
        candidates.append(indices)
        records.append(record)

    # Apply cheap constraints first; integrated coverage is evaluated only on
    # candidates already near the graph/quality frontier.
    for record in records:
        record.update(constraint_record(record, anchor, args))
    shortlist = sorted(range(len(records)), key=lambda index: (
        records[index]["constraint_violation"],
        records[index]["graph_relation_l1"], records[index]["quality"], index))
    if integrated is not None:
        shortlist = shortlist[:args.integrated_coverage_shortlist]
        for index in shortlist:
            add_integrated(records[index], candidates[index], integrated, args)
            records[index].update(constraint_record(records[index], anchor, args))
    feasible = [index for index in shortlist if records[index]["feasible"]]
    if feasible:
        if args.selection_priority == "quality":
            best = min(feasible, key=lambda index: (
                records[index]["quality"], records[index]["graph_relation_l1"], index))
        else:
            best = min(feasible, key=lambda index: (
                records[index]["graph_relation_l1"], records[index]["quality"], index))
    else:
        best = min(shortlist, key=lambda index: (
            records[index]["constraint_violation"],
            records[index]["graph_relation_l1"], records[index]["quality"], index))
    return candidates[best], {
        "anchor": anchor,
        "best": records[best],
        "feasible_draws": len(feasible),
        "evaluated_draws": len(shortlist),
        "swap_count": swap_count,
        "swap_fraction_realized": swap_count / args.count,
        "activation_skipped": False,
        "activation_reason": "graph_deficit_above_threshold",
        "direction_records": direction_records,
        "add_score_summary": {"min": float(add_score.min()), "max": float(add_score.max()),
                              "mean": float(add_score.mean()), "std": float(add_score.std())},
        "remove_score_summary": {"min": float(remove_score.min()), "max": float(remove_score.max()),
                                 "mean": float(remove_score.mean()), "std": float(remove_score.std())},
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
    parser.add_argument("--graph-support-tolerance", type=float, default=.005)
    parser.add_argument("--graph-hard-tolerance", type=float, default=.005)
    parser.add_argument("--graph-constraint-tolerance", type=float, default=1e-8)
    parser.add_argument("--required-graph-gain", type=float, default=.001)
    parser.add_argument("--graph-activation-threshold", type=float, default=0.0)
    parser.add_argument("--quality-relative-tolerance", type=float, required=True)
    parser.add_argument("--quality-absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--swap-fraction", type=float, required=True)
    parser.add_argument("--swap-temperature", type=float, default=.75)
    parser.add_argument("--uniform-mix", type=float, default=.10)
    parser.add_argument("--coverage-boost", type=float, default=2.0)
    parser.add_argument("--hard-boost", type=float, default=2.0)
    parser.add_argument("--selection-priority", choices=("graph", "quality"), default="graph")
    parser.add_argument("--geometric-coverage-weight", type=float, default=0.0)
    parser.add_argument("--geometric-coverage-k", type=int, default=16)
    parser.add_argument("--geometric-coverage-tolerance", type=float, default=0.0)
    parser.add_argument("--integrated-coverage-weight", type=float, default=0.0)
    parser.add_argument("--integrated-coverage-tolerance", type=float, default=.005)
    parser.add_argument("--integrated-coverage-shortlist", type=int, default=8)
    parser.add_argument("--integrated-coverage-folds", type=int, default=2)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if (not 0 <= args.required_graph_gain < 1
            or args.graph_activation_threshold < 0
            or args.quality_relative_tolerance < 0
            or not 0 < args.swap_fraction < 1
            or not 0 < args.swap_temperature
            or not 0 <= args.uniform_mix < 1
            or args.draws <= 0 or args.count <= 0):
        raise ValueError("invalid proximal-repair hyperparameters")

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
    groups = build_groups(
        train, pool, info, args.bins, args.pair_bins,
        args.max_pairs, args.max_pairs > 0, args.target_pair_quota)
    geometry = (
        build_geometry_coverage(train, pool, info, args.count, args.geometric_coverage_k)
        if args.geometric_coverage_weight > 0 else None)
    integrated = (
        build_integrated_coverage(
            train, pool, info, args.count, args.seed, args.integrated_coverage_folds)
        if args.integrated_coverage_weight > 0 else None)
    selected, selection = repair(
        anchor_indices, factors, groups, args, geometry, integrated)

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
        "selection": selection,
        "geometry_reference": geometry.reference_score if geometry else None,
        "integrated_reference": integrated.reference_score if integrated else None,
        "integrated_fold_seeds": list(integrated.fold_seeds) if integrated else [],
    }
    (args.output / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n")
    print(args.output / "selected.csv")
    print(json.dumps({
        "feasible": selection["best"]["feasible"],
        "feasible_draws": selection["feasible_draws"],
        "swap_count": selection["swap_count"],
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
