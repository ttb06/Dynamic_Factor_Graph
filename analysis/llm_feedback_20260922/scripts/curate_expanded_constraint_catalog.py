#!/usr/bin/env python3
"""Curate a role-separated constraint catalog from the frozen unified graph.

The catalog deliberately separates evidence from authority:

* ``actuate`` edges may propose row edits during reverse diffusion;
* ``score_only`` edges may rank lookahead branches but never edit a cell;
* ``monitor`` edges are reported for coverage/calibration only.

Selection uses discovery/counterexample/calibration partitions only.  Held-out
statistics are copied into the post-hoc report, never used in a gate.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


SUPPORTED = {"==", "!=", ">", ">=", "<", "<="}
PARTITIONS = ("discovery", "counterexample", "calibration")

# These are semantic aliases in the non-negative Shoppers schema.  They are
# kept explicit so the audit can distinguish deduplication from rejection.
SEMANTIC_ALIASES = {
    "emp_003": "emp_001",  # duration >= threshold -> count > 0 is subsumed
    "emp_008": "emp_007",  # SpecialDay > 0 and >= .2 are identical in data
    "emp_010": "seed_000",
    "emp_011": "seed_000",  # duration <= 0 is duration == 0 in this schema
    "emp_013": "emp_012",   # count <= 0 is count == 0 in this schema
    "emp_029": "emp_028",
}


def finite(value, default=math.inf) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def atom_key(atom: dict) -> tuple:
    value = atom.get("right_value")
    if atom.get("right_kind") == "literal" and not isinstance(value, bool):
        try:
            numeric = float(value)
            value = round(numeric, 12) if math.isfinite(numeric) else str(value)
        except (TypeError, ValueError):
            value = str(value).strip()
    return (
        atom.get("left_column"), atom.get("operator"), atom.get("right_kind"),
        atom.get("right_column", ""), value,
    )


def executable(edge: dict) -> bool:
    atoms = edge.get("applicability", []) + edge.get("requirements", [])
    atoms += [a for group in edge.get("exceptions", []) for a in group.get("all_of", [])]
    return bool(edge.get("requirements")) and all(a.get("operator") in SUPPORTED for a in atoms)


def split_spread(edge: dict) -> float:
    rates = [finite(edge.get(part, {}).get("violation_rate"), math.nan) for part in PARTITIONS]
    rates = [x for x in rates if math.isfinite(x)]
    return max(rates) - min(rates) if rates else math.inf


def exact_program_key(edge: dict) -> tuple:
    return (
        tuple(sorted(atom_key(a) for a in edge.get("applicability", []))),
        tuple(sorted(atom_key(a) for a in edge.get("requirements", []))),
        tuple(sorted(tuple(sorted(atom_key(a) for a in g.get("all_of", [])))
                     for g in edge.get("exceptions", []))),
    )


def core_program_key(edge: dict) -> tuple:
    """Program identity without exception clauses.

    Two LLM proposals that differ only by an exception must not receive two
    independent actuation votes.  The higher-confidence representative is
    retained and the other remains documented in the decision audit.
    """
    return (
        tuple(sorted(atom_key(a) for a in edge.get("applicability", []))),
        tuple(sorted(atom_key(a) for a in edge.get("requirements", []))),
    )


def exception_masks_a_requirement_failure(edge: dict) -> bool:
    """Detect exceptions that directly name a failing requirement value.

    Such an exception makes the apparent zero-violation evidence partly or
    wholly tautological.  It must not receive actuation authority without a
    separate audit.
    """
    for requirement in edge.get("requirements", []):
        if requirement.get("right_kind") != "literal":
            continue
        for group in edge.get("exceptions", []):
            for atom in group.get("all_of", []):
                if (atom.get("left_column") != requirement.get("left_column") or
                        atom.get("right_kind") != "literal"):
                    continue
                try:
                    value = float(atom.get("right_value"))
                    target = float(requirement.get("right_value"))
                except (TypeError, ValueError):
                    value, target = str(atom.get("right_value")), str(requirement.get("right_value"))
                op = requirement.get("operator")
                fails = ((op == ">" and value <= target) or (op == ">=" and value < target) or
                         (op == "<" and value >= target) or (op == "<=" and value > target) or
                         (op == "==" and value != target) or (op == "!=" and value == target))
                if atom.get("operator") == "==" and fails:
                    return True
    return False


def assign(edge: dict, role: str, rationale: str) -> dict:
    item = dict(edge)
    item["usage_role"] = role
    item["role_rationale"] = rationale
    item["selection_weight"] = {"actuate": 100.0, "score_only": 10.0, "monitor": 0.0}[role]
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="shoppers")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--max-score-edges", type=int, default=8)
    parser.add_argument("--max-monitor-edges", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.graph.read_text())

    funnel = Counter(source=len(source)); decisions = []
    actuate, score_pool, monitor_pool = [], [], []
    for edge in source:
        edge_id = edge.get("candidate_id")
        support = int(edge.get("calibration", {}).get("support", 0))
        rate = finite(edge.get("calibration", {}).get("violation_rate"))
        lift = finite(edge.get("calibration", {}).get("lift"), 0.0)
        spread = split_spread(edge)
        source_kind = edge.get("source")
        klass = edge.get("classification")
        reason = None
        if not executable(edge):
            reason = "not_executable"
        elif support < args.min_support:
            reason = "low_calibration_support"
        elif source_kind != "empirical_mining" and klass in {"hard", "near_hard"}:
            if exception_masks_a_requirement_failure(edge):
                reason = "exception_directly_masks_requirement_failure"
            elif finite(edge.get("judge_confidence"), 0.0) < .65:
                reason = "low_semantic_confidence"
            elif rate > .08:
                reason = "too_many_calibration_exceptions_for_actuation"
            else:
                actuate.append(assign(edge, "actuate", "semantically reviewed hard/near-hard executable edge"))
                funnel["actuate_candidates"] += 1
        elif source_kind == "empirical_mining" and klass in {"hard", "near_hard"}:
            if rate > .08 or spread > .05 or lift < 1.10:
                reason = "unstable_or_weak_empirical_relation"
            else:
                score_pool.append(edge)
                funnel["score_candidates"] += 1
        else:
            if support >= 200 and rate <= .20 and spread <= .06 and lift >= 1.40:
                monitor_pool.append(edge)
                funnel["monitor_candidates"] += 1
            else:
                reason = "soft_relation_below_monitor_evidence"
        if reason:
            decisions.append({"candidate_id": edge_id, "decision": "exclude", "reason": reason})

    # Actuator duplicates previously survived curation and inflated graph size
    # without adding independent information.  Keep one representative per
    # semantic core before assigning any score-only edges.
    actuate.sort(key=lambda e: (
        -finite(e.get("judge_confidence"), 0.0),
        len(e.get("exceptions", [])),
        finite(e.get("calibration", {}).get("violation_rate")),
        -int(e.get("calibration", {}).get("support", 0)),
    ))
    deduplicated_actuate, seen_exact, seen_core = [], {}, {}
    for edge in actuate:
        exact_key, core_key = exact_program_key(edge), core_program_key(edge)
        duplicate = seen_exact.get(exact_key) or seen_core.get(core_key)
        if duplicate:
            decisions.append({"candidate_id": edge["candidate_id"], "decision": "deduplicate",
                              "reason": f"actuator_semantic_core_duplicate_of:{duplicate}"})
            continue
        deduplicated_actuate.append(edge)
        seen_exact[exact_key] = edge["candidate_id"]
        seen_core[core_key] = edge["candidate_id"]
    funnel["actuate_deduplicated"] = len(actuate)-len(deduplicated_actuate)
    actuate = deduplicated_actuate

    # Exact duplicates are removed automatically.  A small set of schema-aware
    # aliases handles mathematically equivalent predicates on non-negative data.
    kept_ids = {e["candidate_id"] for e in actuate}
    exact = {exact_program_key(e): e["candidate_id"] for e in actuate}
    selected_score = []
    score_pool.sort(key=lambda e: (
        0 if e["classification"] == "hard" else 1,
        finite(e["calibration"]["violation_rate"]),
        -int(e["calibration"]["support"]),
    ))
    for edge in score_pool:
        edge_id = edge["candidate_id"]
        # These aliases depend on non-negativity and the exact Shoppers
        # schema.  Reusing candidate IDs on another dataset must not trigger
        # a semantic deduplication.
        duplicate = SEMANTIC_ALIASES.get(edge_id) if args.dataset == "shoppers" else None
        duplicate = duplicate or exact.get(exact_program_key(edge))
        if duplicate and duplicate in kept_ids:
            decisions.append({"candidate_id": edge_id, "decision": "deduplicate", "reason": f"subsumed_by:{duplicate}"})
            continue
        if len(selected_score) >= args.max_score_edges:
            decisions.append({"candidate_id": edge_id, "decision": "reserve", "reason": "score_role_budget"})
            continue
        item = assign(edge, "score_only", "stable empirical edge used as a non-actuating branch reward")
        selected_score.append(item); kept_ids.add(edge_id); exact[exact_program_key(edge)] = edge_id

    selected_monitor = []
    monitor_pool.sort(key=lambda e: (-finite(e["calibration"].get("lift"), 0.0),
                                     finite(e["calibration"].get("violation_rate"))))
    endpoint_patterns = set()
    for edge in monitor_pool:
        endpoint = (tuple(a.get("left_column") for a in edge.get("applicability", [])),
                    tuple(a.get("left_column") for a in edge.get("requirements", [])))
        if endpoint in endpoint_patterns or len(selected_monitor) >= args.max_monitor_edges:
            decisions.append({"candidate_id": edge["candidate_id"], "decision": "reserve", "reason": "monitor_diversity_budget"})
            continue
        selected_monitor.append(assign(edge, "monitor", "stable probabilistic relation retained for diagnostics only"))
        endpoint_patterns.add(endpoint)

    catalog = actuate + selected_score + selected_monitor
    for edge in catalog:
        decisions.append({"candidate_id": edge["candidate_id"], "decision": edge["usage_role"],
                          "reason": edge["role_rationale"]})
    role_counts = Counter(e["usage_role"] for e in catalog)
    summary = {
        "source_edges": len(source), "catalog_edges": len(catalog), **dict(role_counts),
        "sampler_scoring_edges": role_counts["actuate"] + role_counts["score_only"],
        "selection_uses_heldout": False,
    }
    (args.output / "constraint_catalog.json").write_text(json.dumps(catalog, indent=2))
    (args.output / "curation_decisions.json").write_text(json.dumps(decisions, indent=2))
    (args.output / "summary.json").write_text(json.dumps({**summary, "funnel": dict(funnel)}, indent=2))
    lines = ["# Expanded role-separated constraint catalog", "", f"Summary: `{summary}`", "",
             "Held-out statistics were not read by any selection gate. They remain post-hoc evaluation fields.", "",
             "| ID | Usage role | Class | Calibration support | Calibration violation | Split spread | Description |",
             "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for edge in catalog:
        lines.append(f"| {edge['candidate_id']} | {edge['usage_role']} | {edge['classification']} | "
                     f"{edge['calibration']['support']} | {edge['calibration']['violation_rate']:.4f} | "
                     f"{split_spread(edge):.4f} | {edge['description']} |")
    lines += ["", "The graph is expanded for state-dependent scoring without granting empirical associations permission to edit cells."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
