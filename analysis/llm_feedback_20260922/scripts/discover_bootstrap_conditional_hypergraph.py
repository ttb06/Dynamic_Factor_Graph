#!/usr/bin/env python3
"""Discover an API-free bootstrap-stable conditional constraint hypergraph.

The earlier selector ranked single implications from one three-way split and
then forced a DAG.  This method treats graph construction as stability
selection: candidate relations are re-evaluated on bootstrap samples, their
uncertainty is retained, and conjunctive applicability hyperedges are admitted
only when they improve consequent predictability beyond either parent rule.
Selection rewards evidence, stability, and graph coverage; cycles are allowed
because they represent statistical compatibility, not causal direction.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def typed_atom(atom: dict) -> dict:
    if "left_column" in atom:
        return {"left_column": atom["left_column"], "operator": atom["operator"],
                "right_kind": atom.get("right_kind", "literal"),
                "right_column": atom.get("right_column", ""),
                "right_value": atom.get("right_value", 0),
                "tolerance": float(atom.get("tolerance", 0))}
    return {"left_column": atom["column"], "operator": atom["operator"],
            "right_kind": "literal", "right_column": "",
            "right_value": atom.get("value", 0), "tolerance": 0.0}


def signature(atoms: list[dict]) -> str:
    normalized = [typed_atom(atom) for atom in atoms]
    normalized.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def edge_description(applicability: list[dict], requirements: list[dict]) -> str:
    def render(atom):
        right = atom["right_column"] if atom["right_kind"] == "column" else repr(atom["right_value"])
        return f"{atom['left_column']} {atom['operator']} {right}"
    left = " AND ".join(render(atom) for atom in applicability) if applicability else "TRUE"
    return f"IF {left} THEN " + " AND ".join(render(atom) for atom in requirements)


def edge_columns(edge: dict) -> tuple[set[str], set[str]]:
    source = {atom["left_column"] for atom in edge.get("applicability", [])}
    target = {atom["left_column"] for atom in edge.get("requirements", [])}
    return source, target


def bootstrap_statistics(frame: pd.DataFrame, candidate: dict, verify, rng,
                         bootstraps: int) -> dict:
    rates, supports, lifts = [], [], []
    for _ in range(bootstraps):
        sampled = frame.iloc[rng.integers(0, len(frame), len(frame))]
        stats = verify(sampled, candidate)
        if int(stats.get("support", 0)):
            rates.append(float(stats["violation_rate"])); supports.append(int(stats["support"]))
            lifts.append(float(stats.get("lift", 0.0)))
    if not rates:
        return {"replicates": 0, "rate_mean": math.nan, "rate_std": math.nan,
                "rate_q05": math.nan, "rate_q95": math.nan, "support_q05": 0,
                "lift_mean": math.nan, "stability_probability": 0.0}
    array = np.asarray(rates)
    return {"replicates": len(rates), "rate_mean": float(array.mean()),
            "rate_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "rate_q05": float(np.quantile(array, .05)), "rate_q95": float(np.quantile(array, .95)),
            "support_q05": int(np.quantile(supports, .05)), "lift_mean": float(np.mean(lifts)),
            "stability_probability": float(np.mean(array <= .25))}


def classification(q95_rate: float) -> str:
    if q95_rate <= .01: return "hard"
    if q95_rate <= .10: return "near_hard"
    return "soft"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=32); parser.add_argument("--seed", type=int, default=20261430)
    parser.add_argument("--max-edges", type=int, default=16); parser.add_argument("--max-hyperedges", type=int, default=6)
    parser.add_argument("--min-hyperedges", type=int, default=2,
                        help="Reserve budget for higher-order rules that pass the synergy test.")
    parser.add_argument("--min-support", type=int, default=150); parser.add_argument("--min-lift", type=float, default=1.15)
    parser.add_argument("--max-rate", type=float, default=.25); parser.add_argument("--max-rate-std", type=float, default=.05)
    parser.add_argument("--min-synergy", type=float, default=.01)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    scripts = args.repo / "analysis/llm_feedback_20260922/scripts"; sys.path.insert(0, str(scripts))
    from knowledge_graph_protocol import verify
    from typed_dynamic_graph_guidance import edge_masks

    train = pd.read_csv(args.repo / "data" / args.dataset / "train.csv")
    source = json.loads(args.source.read_text()); rng = np.random.default_rng(args.seed)
    candidates, audit = [], []
    for raw in source:
        applicability = [typed_atom(atom) for atom in raw.get("applicability", [])]
        requirements = [typed_atom(atom) for atom in raw.get("requirements", [])]
        if not applicability or not requirements:
            audit.append({"candidate_id": raw.get("candidate_id"), "decision": "exclude",
                          "reason": "empty_applicability_or_requirement"}); continue
        candidate = copy.deepcopy(raw); candidate["applicability"] = applicability
        candidate["requirements"] = requirements; candidate["exceptions"] = []
        full = verify(train, candidate)
        active_mask, _, bad_mask = edge_masks(train, candidate)
        truth_signature = hashlib.sha1(
            np.packbits(active_mask).tobytes() + np.packbits(bad_mask).tobytes()
        ).hexdigest()
        bootstrap = bootstrap_statistics(train, candidate, verify, rng, args.bootstraps)
        reason = None
        if int(full.get("support", 0)) < args.min_support or bootstrap["support_q05"] < args.min_support // 2:
            reason = "insufficient_support"
        elif float(full.get("lift", 0.0)) < args.min_lift:
            reason = "insufficient_lift"
        elif bootstrap["rate_q95"] > args.max_rate:
            reason = "high_upper_violation_rate"
        elif bootstrap["rate_std"] > args.max_rate_std:
            reason = "bootstrap_instability"
        if reason:
            audit.append({"candidate_id": raw.get("candidate_id"), "decision": "exclude",
                          "reason": reason, "full": full, "bootstrap": bootstrap}); continue
        confidence_gain = max(0.0, (1.0 - float(full["violation_rate"])) - float(full.get("base_rate", 0)))
        score = math.log1p(int(full["support"])) * confidence_gain * max(bootstrap["stability_probability"], .01) \
            / (1.0 + 5.0 * bootstrap["rate_std"])
        candidates.append({"kind": "pairwise", "raw": candidate, "full": full,
                           "bootstrap": bootstrap, "score": score,
                           "truth_signature": truth_signature})

    # Rules that differ syntactically but induce the same applicability and
    # violation masks (e.g. x==0 and x<=0 on a non-negative feature) are one
    # empirical edge, not evidence for additional graph density.
    unique_candidates = {}
    for candidate in candidates:
        key = candidate["truth_signature"]
        if key not in unique_candidates or candidate["score"] > unique_candidates[key]["score"]:
            unique_candidates[key] = candidate
    redundant_pairwise = len(candidates) - len(unique_candidates)
    candidates = list(unique_candidates.values())

    # Higher-order applicability is proposed only within a shared consequent.
    # This makes the incremental test meaningful and avoids combinatorial rule
    # enumeration over unrelated columns.
    hyperedges = []
    by_requirement: dict[str, list[dict]] = {}
    for candidate in candidates:
        by_requirement.setdefault(signature(candidate["raw"]["requirements"]), []).append(candidate)
    for group in by_requirement.values():
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left, right = group[left_index], group[right_index]
                combined = left["raw"]["applicability"] + right["raw"]["applicability"]
                unique = {json.dumps(atom, sort_keys=True): atom for atom in combined}
                combined = list(unique.values())
                if len(combined) < 2 or len(combined) > 3: continue
                # Two different equality literals on one column cannot be a
                # useful conjunction.
                equalities = [(atom["left_column"], str(atom["right_value"])) for atom in combined
                              if atom["operator"] == "==" and atom["right_kind"] == "literal"]
                if any(len({value for column, value in equalities if column == current}) > 1
                       for current, _ in equalities):
                    continue
                raw = copy.deepcopy(left["raw"]); raw["applicability"] = combined
                raw["candidate_id"] = f"hyper_{len(hyperedges):03d}"
                raw["description"] = edge_description(combined, raw["requirements"])
                full = verify(train, raw)
                if int(full.get("support", 0)) < args.min_support: continue
                parent_confidence = max(1.0 - float(left["full"]["violation_rate"]),
                                        1.0 - float(right["full"]["violation_rate"]))
                synergy = (1.0 - float(full["violation_rate"])) - parent_confidence
                if synergy < args.min_synergy: continue
                bootstrap = bootstrap_statistics(train, raw, verify, rng, args.bootstraps)
                if bootstrap["rate_q95"] > args.max_rate or bootstrap["rate_std"] > args.max_rate_std:
                    continue
                score = math.log1p(int(full["support"])) * synergy * (1.0 + len(combined)) \
                    / (1.0 + 5.0 * bootstrap["rate_std"])
                hyperedges.append({"kind": "conjunctive", "raw": raw, "full": full,
                                   "bootstrap": bootstrap, "score": score,
                                   "synergy": synergy,
                                   "parents": [left["raw"].get("candidate_id"), right["raw"].get("candidate_id")]})

    pool = sorted(candidates, key=lambda item: -item["score"])
    hyperedges = sorted(hyperedges, key=lambda item: -item["score"])[:args.max_hyperedges]
    selected, covered, degree = [], set(), Counter(); hyper_count = 0
    # Order-balanced stability selection.  Pairwise and conjunctive scores are
    # not numerically comparable because synergy is an incremental effect;
    # reserve a small, explicit part of the graph budget for validated motifs.
    reserved = hyperedges[:min(args.min_hyperedges, len(hyperedges), args.max_edges)]
    for choice in reserved:
        source_columns, target_columns = edge_columns(choice["raw"]); nodes = source_columns | target_columns
        covered |= nodes
        for node in nodes: degree[node] += 1
        hyper_count += 1
        selected.append(choice)
    pool.extend(hyperedges[len(reserved):])
    while pool and len(selected) < args.max_edges:
        ranked = []
        for item in pool:
            source_columns, target_columns = edge_columns(item["raw"])
            nodes = source_columns | target_columns
            novelty = len(nodes - covered) / max(len(nodes), 1)
            overload = sum(max(0, degree[node] - 3) for node in nodes)
            adjusted = item["score"] * (1.0 + .25 * novelty) / (1.0 + .15 * overload)
            if item["kind"] == "conjunctive" and hyper_count >= args.max_hyperedges:
                adjusted = -math.inf
            ranked.append((adjusted, item))
        _, choice = max(ranked, key=lambda pair: pair[0]); pool.remove(choice)
        if not math.isfinite(max(ranked, key=lambda pair: pair[0])[0]): break
        source_columns, target_columns = edge_columns(choice["raw"]); nodes = source_columns | target_columns
        covered |= nodes
        for node in nodes: degree[node] += 1
        if choice["kind"] == "conjunctive": hyper_count += 1
        selected.append(choice)
    # Convert selected candidate objects to the public typed graph format.
    public_selected = []
    for rank, choice in enumerate(selected, start=1):
        raw, full, bootstrap = choice["raw"], choice["full"], choice["bootstrap"]
        edge = {
            "description": edge_description(raw["applicability"], raw["requirements"]),
            "applicability": raw["applicability"], "requirements": raw["requirements"], "exceptions": [],
            "relation_family": raw.get("relation_family", "distributional"),
            "hardness_prior": classification(bootstrap["rate_q95"]),
            "evidence_basis": "empirical_signal", "rationale": "bootstrap-stable train-only conditional relation",
            "directionality_reason": "conditional confidence and lift are evaluated in the stated direction",
            "plausible_counterexample": "retained as calibrated empirical violation mass",
            "confidence": 1.0 - float(full["violation_rate"]),
            "candidate_id": f"bscg_{rank-1:03d}", "source": "bootstrap_conditional_graph",
            "discovery": raw.get("discovery", full), "counterexample": raw.get("counterexample", full),
            "calibration": {**raw.get("calibration", full), **full},
            "classification": classification(bootstrap["rate_q95"]),
            "provenance": ["train_only_empirical_mining", "bootstrap_stability_selection", choice["kind"]],
            "usage_role": "actuate", "selection_score": choice["score"],
            "bootstrap_uncertainty": bootstrap, "edge_order": len(raw["applicability"]),
        }
        if choice["kind"] == "conjunctive":
            edge["conditional_synergy"] = choice["synergy"]; edge["parent_candidates"] = choice["parents"]
        public_selected.append(edge)
        audit.append({"candidate_id": raw.get("candidate_id"), "decision": "select",
                      "kind": choice["kind"], "score": choice["score"], "rank": rank})

    (args.output / "graph.json").write_text(json.dumps(public_selected, indent=2))
    (args.output / "selection_audit.json").write_text(json.dumps(audit, indent=2))
    metadata = {"method": "bootstrap_stable_conditional_hypergraph", "llm_used": False, "api_calls": 0,
                "dataset": args.dataset, "train_rows": len(train), "source_candidates": len(source),
                "eligible_pairwise": len(candidates), "redundant_pairwise_removed": redundant_pairwise,
                "eligible_conjunctive": len(hyperedges),
                "selected_edges": len(public_selected), "selected_hyperedges": hyper_count,
                "covered_nodes": len(covered), "bootstraps": args.bootstraps, "seed": args.seed,
                "heldout_used_for_selection": False,
                "filters": {"min_support": args.min_support, "min_lift": args.min_lift,
                            "max_rate": args.max_rate, "max_rate_std": args.max_rate_std,
                            "min_synergy": args.min_synergy}}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    for edge in public_selected: print(edge["candidate_id"], edge["description"], edge["selection_score"])


if __name__ == "__main__": main()
