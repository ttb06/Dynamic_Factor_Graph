#!/usr/bin/env python3
"""Graph-consistent, train-only subset selection for tabular diffusion pools.

The selector reuses the verified factor graph that actuated the dynamic sampler.
Each factor defines a three-state train/pool statistic: not applicable,
applicable-and-satisfied, or applicable-and-violated. Candidate subsets are
ranked by the generic mixed-type moment objective plus weighted graph-relation,
factor-support coverage, and optional hard-edge admissibility terms.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from balanced_particle_resampling import (
    MomentGroup,
    build_geometry_coverage,
    build_groups,
    build_integrated_coverage,
    canonicalize,
    moment_loss,
    rake,
    sha256,
)
from typed_dynamic_graph_guidance import edge_masks


@dataclass(frozen=True)
class GraphFactor:
    candidate_id: str
    train_states: np.ndarray
    pool_states: np.ndarray
    weight: float
    hard: bool
    train_support: float
    train_violation_rate: float
    usage_role: str
    confidence: float


def state_codes(frame: pd.DataFrame, edge: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active, satisfied, bad = edge_masks(frame, edge)
    codes = np.zeros(len(frame), dtype=np.int64)
    codes[active & satisfied] = 1
    codes[bad] = 2
    return codes, active, bad


def edge_confidence(edge: dict) -> float:
    values = [edge.get("confidence"), edge.get("judge_confidence")]
    for section in ("calibration", "counterexample", "discovery"):
        if isinstance(edge.get(section), dict):
            values.append(edge[section].get("confidence"))
    numeric = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            numeric.append(numeric_value)
    return float(np.clip(np.mean(numeric), 0.05, 1.0)) if numeric else 0.5


def edge_weight(edge: dict) -> tuple[float, float]:
    role = edge.get("usage_role", "actuate")
    role_weight = {"actuate": 1.0, "score_only": 0.50, "monitor": 0.25}.get(role, 0.25)
    confidence = edge_confidence(edge)
    stability = float(edge.get("bootstrap_uncertainty", {}).get("stability_probability", 1.0))
    raw_score = float(edge.get("selection_score", edge.get("selection_weight", 1.0)))
    # Scores come from heterogeneous graph builders (roughly 1--6 versus 100).
    # A bounded logarithmic multiplier preserves ordering without allowing a
    # builder-specific scale to dominate the semantic objective.
    score_multiplier = 1.0 + 0.10 * math.log1p(max(raw_score, 0.0))
    return role_weight * confidence * np.clip(stability, 0.1, 1.0) * score_multiplier, confidence


def load_graph_factors(train: pd.DataFrame, pool: pd.DataFrame, graph_path: Path,
                       edge_ids: set[str], roles: set[str], min_support: int) -> tuple[list[GraphFactor], list[dict]]:
    graph = json.loads(graph_path.read_text())
    if not isinstance(graph, list):
        raise TypeError("graph root must be a list of factor edges")
    factors: list[GraphFactor] = []
    skipped: list[dict] = []
    for edge in graph:
        candidate_id = str(edge.get("candidate_id", ""))
        role = str(edge.get("usage_role", "actuate"))
        if edge_ids and candidate_id not in edge_ids:
            continue
        if role not in roles:
            continue
        try:
            train_codes, train_active, train_bad = state_codes(train, edge)
            pool_codes, _, _ = state_codes(pool, edge)
        except (KeyError, TypeError, ValueError) as error:
            skipped.append({"candidate_id": candidate_id, "reason": str(error)})
            continue
        support = int(train_active.sum())
        if support < min_support:
            skipped.append({"candidate_id": candidate_id, "reason": f"support={support}<{min_support}"})
            continue
        weight, confidence = edge_weight(edge)
        hard = str(edge.get("classification", edge.get("hardness_prior", "soft"))) == "hard"
        factors.append(GraphFactor(
            candidate_id=candidate_id,
            train_states=train_codes,
            pool_states=pool_codes,
            weight=float(weight),
            hard=hard,
            train_support=float(train_active.mean()),
            train_violation_rate=float(train_bad.sum() / max(train_active.sum(), 1)),
            usage_role=role,
            confidence=confidence,
        ))
    if not factors:
        raise ValueError("no eligible graph factors")
    scale = float(np.mean([factor.weight for factor in factors]))
    factors = [GraphFactor(**{**factor.__dict__, "weight": factor.weight / max(scale, 1e-12)})
               for factor in factors]
    return factors, skipped


def smoothed_distribution(codes: np.ndarray, cardinality: int = 3) -> np.ndarray:
    counts = np.bincount(codes, minlength=cardinality).astype(float) + 0.5
    return counts / counts.sum()


def graph_loss(indices: np.ndarray, factors: Sequence[GraphFactor],
               support_tolerance: float, hard_tolerance: float) -> tuple[float, float, float, list[dict]]:
    relation_numerator = 0.0
    coverage_numerator = 0.0
    hard_numerator = 0.0
    total_weight = 0.0
    hard_weight = 0.0
    records: list[dict] = []
    for factor in factors:
        target = smoothed_distribution(factor.train_states)
        observed = smoothed_distribution(factor.pool_states[indices])
        relation = float(np.abs(target - observed).sum())
        observed_support = float(np.mean(factor.pool_states[indices] > 0))
        support_deficit = max(
            factor.train_support - support_tolerance - observed_support, 0.0)
        coverage = support_deficit / math.sqrt(max(factor.train_support, 1e-8))
        active = factor.pool_states[indices] > 0
        violation = factor.pool_states[indices] == 2
        observed_violation = float(violation.sum() / max(active.sum(), 1))
        hard_excess = (
            max(observed_violation - factor.train_violation_rate - hard_tolerance, 0.0)
            if factor.hard else 0.0)
        relation_numerator += factor.weight * relation
        coverage_numerator += factor.weight * coverage
        total_weight += factor.weight
        if factor.hard:
            hard_numerator += factor.weight * hard_excess
            hard_weight += factor.weight
        records.append({
            "candidate_id": factor.candidate_id,
            "weight": factor.weight,
            "hard": factor.hard,
            "relation_l1": relation,
            "train_support": factor.train_support,
            "selected_support": observed_support,
            "support_deficit": support_deficit,
            "train_violation_rate": factor.train_violation_rate,
            "selected_violation_rate": observed_violation,
            "hard_excess": hard_excess,
        })
    relation_mean = relation_numerator / max(total_weight, 1e-12)
    coverage_mean = coverage_numerator / max(total_weight, 1e-12)
    hard_mean = hard_numerator / max(hard_weight, 1e-12) if hard_weight else 0.0
    return float(relation_mean), float(coverage_mean), float(hard_mean), records


def consensus_weights(pool_rows: int, count: int, paths: Sequence[Path],
                      prior_weights: Sequence[float], strength: float,
                      clip: float) -> tuple[np.ndarray, dict]:
    if not paths:
        return np.ones(pool_rows, dtype=float), {"enabled": False, "priors": []}
    if len(paths) != len(prior_weights) or any(value <= 0 for value in prior_weights):
        raise ValueError("prior paths/weights must be matched and positive")
    votes = np.zeros(pool_rows, dtype=float)
    records = []
    for path, weight in zip(paths, prior_weights):
        frame = pd.read_csv(path)
        if list(frame.columns) != ["pool_index"]:
            raise ValueError(f"invalid prior-index schema: {path}")
        indices = frame["pool_index"].to_numpy(np.int64)
        if (len(indices) != count or len(np.unique(indices)) != count or
                indices.min(initial=0) < 0 or indices.max(initial=-1) >= pool_rows):
            raise ValueError(f"invalid prior indices: {path}")
        votes[indices] += weight
        records.append({"path": str(path), "sha256": sha256(path), "weight": weight})
    votes /= float(sum(prior_weights))
    centered = votes - float(votes.mean())
    weights = np.exp(np.clip(strength * centered, -12.0, 12.0))
    weights /= max(weights.mean(), 1e-12)
    weights = np.clip(weights, 1.0 / clip, clip)
    return weights, {
        "enabled": True, "priors": records,
        "vote_min": float(votes.min()), "vote_max": float(votes.max()),
        "vote_mean": float(votes.mean()),
    }


def graph_groups(factors: Sequence[GraphFactor]) -> list[MomentGroup]:
    return [MomentGroup(
        name=f"graph::{factor.candidate_id}",
        train_codes=factor.train_states,
        pool_codes=factor.pool_states,
        cardinality=3,
        family="graph",
    ) for factor in factors]


def candidate_objective(indices: np.ndarray, generic_groups: Sequence[MomentGroup],
                        factors: Sequence[GraphFactor], args: argparse.Namespace,
                        geometry) -> dict:
    generic, univariate, pair, rare = moment_loss(
        indices, generic_groups, args.pair_weight, args.coverage_weight)
    relation, graph_coverage, hard_excess, edge_records = graph_loss(
        indices, factors, args.graph_support_tolerance,
        args.graph_hard_tolerance)
    geometry_score = geometry.score(indices, args.pool_rows) if geometry else 0.0
    geometry_deficit = (
        max(geometry.reference_score - args.geometric_coverage_tolerance - geometry_score, 0.0)
        if geometry else 0.0)
    objective = (
        args.generic_weight * generic
        + args.graph_relation_weight * relation
        + args.graph_coverage_weight * graph_coverage
        + args.graph_hard_weight * hard_excess
        + args.geometric_coverage_weight * geometry_deficit
    )
    return {
        "objective": float(objective),
        "generic_objective": float(generic),
        "univariate_l1": float(univariate),
        "pair_l1": float(pair),
        "rare_cell_coverage_loss": float(rare),
        "graph_relation_l1": relation,
        "graph_coverage_gap": graph_coverage,
        "graph_hard_excess": hard_excess,
        "graph_hard_admissible": hard_excess <= 1e-12,
        "geometric_coverage": geometry_score,
        "geometric_coverage_deficit": geometry_deficit,
        "integrated_alpha": None,
        "integrated_coverage": None,
        "integrated_coverage_deficit": None,
        "edge_records": edge_records,
    }


def sample_best(weights: np.ndarray, generic_groups: Sequence[MomentGroup],
                factors: Sequence[GraphFactor], args: argparse.Namespace,
                geometry, integrated) -> tuple[np.ndarray, dict]:
    probabilities = np.power(np.clip(weights, 1e-12, None), args.temperature)
    probabilities /= probabilities.sum()
    probabilities = (1.0 - args.tail_mix) * probabilities + args.tail_mix / len(probabilities)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(args.seed)
    indices_by_draw: list[np.ndarray] = []
    records: list[dict] = []
    for draw in range(args.draws):
        indices = np.sort(rng.choice(
            len(weights), size=args.count, replace=False, p=probabilities))
        record = candidate_objective(indices, generic_groups, factors, args, geometry)
        record["draw"] = draw
        indices_by_draw.append(indices)
        records.append(record)
    eligible = list(range(args.draws))
    if args.graph_hard_admissibility:
        eligible = [index for index in eligible if records[index]["graph_hard_admissible"]]
        if not eligible:
            raise ValueError("no draw satisfies graph hard admissibility")
    if integrated is not None:
        eligible = sorted(eligible, key=lambda index: (records[index]["objective"], index))[
            :args.integrated_coverage_shortlist]
        for index in eligible:
            alpha, beta = integrated.score(indices_by_draw[index])
            deficit = max(
                integrated.reference_score - args.integrated_coverage_tolerance - beta, 0.0)
            records[index]["integrated_alpha"] = alpha
            records[index]["integrated_coverage"] = beta
            records[index]["integrated_coverage_deficit"] = deficit
            records[index]["objective"] += args.integrated_coverage_weight * deficit
    best_draw = min(eligible, key=lambda index: (records[index]["objective"], index))
    uniform_indices = np.sort(rng.choice(len(weights), size=args.count, replace=False))
    uniform = candidate_objective(
        uniform_indices, generic_groups, factors, args, geometry)
    if integrated is not None:
        alpha, beta = integrated.score(uniform_indices)
        deficit = max(
            integrated.reference_score - args.integrated_coverage_tolerance - beta, 0.0)
        uniform["integrated_alpha"] = alpha
        uniform["integrated_coverage"] = beta
        uniform["integrated_coverage_deficit"] = deficit
        uniform["objective"] += args.integrated_coverage_weight * deficit
    return indices_by_draw[best_draw], {
        "best": records[best_draw],
        "uniform_control": uniform,
        "median_objective": float(np.median([record["objective"] for record in records])),
        "effective_sample_size": float(1.0 / np.square(probabilities).sum()),
        "records": records,
        "uniform_indices": uniform_indices,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--backbone-pool", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--include-edge-ids", default="")
    parser.add_argument("--include-roles", default="actuate")
    parser.add_argument("--graph-min-support", type=int, default=30)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--pair-bins", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--target-pair-quota", type=int, default=8)
    parser.add_argument("--pair-weight", type=float, default=1.5)
    parser.add_argument("--coverage-weight", type=float, default=0.30)
    parser.add_argument("--generic-weight", type=float, default=1.0)
    parser.add_argument("--generic-rake-iterations", type=int, default=5)
    parser.add_argument("--generic-rake-damping", type=float, default=0.5)
    parser.add_argument("--graph-rake-iterations", type=int, default=2)
    parser.add_argument("--graph-rake-damping", type=float, default=0.25)
    parser.add_argument("--graph-relation-weight", type=float, default=0.5)
    parser.add_argument("--graph-coverage-weight", type=float, default=1.0)
    parser.add_argument("--graph-hard-weight", type=float, default=10.0)
    parser.add_argument("--graph-support-tolerance", type=float, default=0.005)
    parser.add_argument("--graph-hard-tolerance", type=float, default=0.005)
    parser.add_argument("--graph-hard-admissibility", action="store_true")
    parser.add_argument("--prior-index", action="append", type=Path, default=[])
    parser.add_argument("--prior-weight", action="append", type=float, default=[])
    parser.add_argument("--consensus-strength", type=float, default=4.0)
    parser.add_argument("--geometric-coverage-weight", type=float, default=0.0)
    parser.add_argument("--geometric-coverage-k", type=int, default=16)
    parser.add_argument("--geometric-coverage-tolerance", type=float, default=0.0)
    parser.add_argument("--integrated-coverage-weight", type=float, default=0.0)
    parser.add_argument("--integrated-coverage-tolerance", type=float, default=0.005)
    parser.add_argument("--integrated-coverage-shortlist", type=int, default=8)
    parser.add_argument("--integrated-coverage-folds", type=int, default=2)
    parser.add_argument("--weight-clip", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--tail-mix", type=float, default=0.10)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.count:
        raise ValueError("count must be positive")
    if (args.target_pair_quota < 0 or args.target_pair_quota > args.max_pairs or
            args.generic_weight < 0 or args.graph_relation_weight < 0 or
            args.graph_coverage_weight < 0 or args.graph_hard_weight < 0 or
            not 0 <= args.tail_mix <= 1 or args.temperature <= 0):
        raise ValueError("invalid selection hyperparameters")
    if (args.integrated_coverage_weight > 0 and
            args.integrated_coverage_shortlist > args.draws):
        raise ValueError("integrated shortlist exceeds draws")

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_path = args.repo / "synthetic" / args.dataset / "real.csv"
    train = canonicalize(pd.read_csv(train_path), info)
    pool = canonicalize(pd.read_csv(args.pool), info)
    backbone = canonicalize(pd.read_csv(args.backbone_pool), info)
    if len(pool) != len(backbone) or len(pool) <= args.count:
        raise ValueError("pool/backbone mismatch or no oversampling")
    args.pool_rows = len(pool)
    edge_ids = {value for value in args.include_edge_ids.split(",") if value}
    roles = {value for value in args.include_roles.split(",") if value}
    factors, skipped = load_graph_factors(
        train, pool, args.graph, edge_ids, roles, args.graph_min_support)
    generic_groups = build_groups(
        train, pool, info, args.bins, args.pair_bins,
        args.max_pairs, args.max_pairs > 0, args.target_pair_quota)
    weights, prior_diagnostics = consensus_weights(
        len(pool), args.count, args.prior_index, args.prior_weight,
        args.consensus_strength, args.weight_clip)
    generic_trace: list[dict] = []
    if args.generic_rake_iterations > 0:
        univariate = [group for group in generic_groups if group.family == "univariate"]
        weights, generic_trace = rake(
            weights, univariate, args.generic_rake_iterations,
            args.generic_rake_damping, args.weight_clip)
    graph_trace: list[dict] = []
    if args.graph_rake_iterations > 0:
        weights, graph_trace = rake(
            weights, graph_groups(factors), args.graph_rake_iterations,
            args.graph_rake_damping, args.weight_clip)
    geometry = None
    if args.geometric_coverage_weight > 0:
        geometry = build_geometry_coverage(
            train, pool, info, args.count, args.geometric_coverage_k)
    integrated = None
    if args.integrated_coverage_weight > 0:
        integrated = build_integrated_coverage(
            train, pool, info, args.count, args.seed,
            args.integrated_coverage_folds)
    selected, selection = sample_best(
        weights, generic_groups, factors, args, geometry, integrated)
    uniform = np.asarray(selection.pop("uniform_indices"), dtype=np.int64)
    args.output.mkdir(parents=True)
    pool.iloc[selected].reset_index(drop=True).to_csv(args.output / "selected.csv", index=False)
    pool.iloc[:args.count].reset_index(drop=True).to_csv(args.output / "g_reference.csv", index=False)
    backbone.iloc[:args.count].reset_index(drop=True).to_csv(args.output / "b_reference.csv", index=False)
    pool.iloc[uniform].reset_index(drop=True).to_csv(args.output / "uniform_control.csv", index=False)
    pd.DataFrame({"pool_index": selected}).to_csv(args.output / "selected_indices.csv", index=False)

    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "train_rows": len(train), "pool_rows": len(pool),
        "selected_rows": len(selected),
        "train_sha256": sha256(train_path),
        "pool_sha256": sha256(args.pool),
        "backbone_pool_sha256": sha256(args.backbone_pool),
        "graph_sha256": sha256(args.graph),
        "parameters": {
            key: value for key, value in vars(args).items()
            if key not in {"repo", "pool", "backbone_pool", "graph", "output", "pool_rows"}
        },
        "graph_factors": [{
            "candidate_id": factor.candidate_id,
            "weight": factor.weight, "hard": factor.hard,
            "usage_role": factor.usage_role, "confidence": factor.confidence,
            "train_support": factor.train_support,
            "train_violation_rate": factor.train_violation_rate,
        } for factor in factors],
        "skipped_graph_factors": skipped,
        "generic_groups": [{
            "name": group.name, "family": group.family,
            "cardinality": group.cardinality,
        } for group in generic_groups],
        "prior": prior_diagnostics,
        "generic_rake_trace": generic_trace,
        "graph_rake_trace": graph_trace,
        "selection": selection,
        "geometry_reference": geometry.reference_score if geometry else None,
        "integrated_reference": integrated.reference_score if integrated else None,
        "integrated_fold_seeds": list(integrated.fold_seeds) if integrated else [],
        "weight_summary": {
            "min": float(weights.min()), "max": float(weights.max()),
            "mean": float(weights.mean()), "std": float(weights.std()),
        },
        "no_duplicate_selected_rows": len(np.unique(selected)) == len(selected),
    }
    (args.output / "selection_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(args.output / "selected.csv")
    print(json.dumps({
        "best_objective": selection["best"]["objective"],
        "best_graph_relation_l1": selection["best"]["graph_relation_l1"],
        "best_graph_coverage_gap": selection["best"]["graph_coverage_gap"],
        "best_graph_hard_excess": selection["best"]["graph_hard_excess"],
        "uniform_objective": selection["uniform_control"]["objective"],
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
