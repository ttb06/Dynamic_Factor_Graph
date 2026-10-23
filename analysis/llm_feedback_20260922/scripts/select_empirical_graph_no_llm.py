#!/usr/bin/env python3
"""Select a connected acyclic guidance graph using train statistics only.

This selector intentionally ignores descriptions, LLM decisions, external
knowledge, held-out statistics, and API artifacts.  Candidates must originate
from empirical mining and pass discovery/counterexample/calibration gates.
Greedy selection then maximizes a deterministic association/stability score
while excluding cycles and transitive shortcuts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PARTITIONS = ("discovery", "counterexample", "calibration")


def typed_atom(atom: dict) -> dict:
    if "left_column" in atom:
        return {
            "left_column": atom["left_column"],
            "operator": atom["operator"],
            "right_kind": atom.get("right_kind", "literal"),
            "right_column": atom.get("right_column", ""),
            "right_value": atom.get("right_value", 0),
            "tolerance": float(atom.get("tolerance", 0)),
        }
    return {
        "left_column": atom["column"], "operator": atom["operator"],
        "right_kind": "literal", "right_column": "",
        "right_value": atom.get("value", 0), "tolerance": 0,
    }


def consequent_atom(edge: dict) -> dict | None:
    if edge.get("requirements"):
        return typed_atom(edge["requirements"][0])
    if edge.get("consequent"):
        return typed_atom(edge["consequent"])
    return None


def has_path(graph: dict[str, set[str]], source: str, target: str) -> bool:
    stack, seen = [source], set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node); stack.extend(graph.get(node, ()))
    return False


def machine_description(applicability: list[dict], requirement: dict) -> str:
    def render(atom: dict) -> str:
        right = atom["right_column"] if atom["right_kind"] == "column" else repr(atom["right_value"])
        return f"{atom['left_column']} {atom['operator']} {right}"
    return "IF " + " AND ".join(map(render, applicability)) + " THEN " + render(requirement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-edges", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=300)
    parser.add_argument("--min-lift", type=float, default=1.5)
    parser.add_argument("--max-rate", type=float, default=.25)
    parser.add_argument("--max-spread", type=float, default=.08)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    source = json.loads(args.source.read_text())
    info = json.loads(args.info.read_text())
    names = info["column_names"]
    categorical = {names[int(index)] for index in info["cat_col_idx"]}
    target = {names[int(index)] for index in info["target_col_idx"]}
    candidates, audit = [], []

    for raw in source:
        edge_id = str(raw.get("candidate_id", "unknown"))
        reason = None
        if raw.get("source") != "empirical_mining":
            reason = "not_empirical_mining"
        applicability = [typed_atom(atom) for atom in raw.get("applicability", [])]
        requirement = consequent_atom(raw)
        if not reason and (len(applicability) != 1 or requirement is None):
            reason = "not_single_antecedent_single_consequent"
        if not reason and (requirement["left_column"] not in categorical or
                           requirement["left_column"] in target or
                           requirement["operator"] != "==" or
                           requirement["right_kind"] != "literal"):
            reason = "unsupported_or_target_consequent"
        statistics = [raw.get(part, {}) for part in PARTITIONS]
        rates = [float(item.get("violation_rate", math.inf)) for item in statistics]
        supports = [int(item.get("support", 0)) for item in statistics]
        calibration = raw.get("calibration", {})
        lift = float(calibration.get("lift", 0))
        if not reason and min(supports) < args.min_support:
            reason = "insufficient_partition_support"
        if not reason and (max(rates) > args.max_rate or max(rates) - min(rates) > args.max_spread):
            reason = "weak_or_unstable_relation"
        if not reason and lift < args.min_lift:
            reason = "insufficient_lift"
        if reason:
            audit.append({"candidate_id": edge_id, "decision": "exclude", "reason": reason})
            continue

        conditional = 1.0 - rates[-1]
        base_rate = float(calibration.get("base_rate", 0))
        stability = 1.0 - (max(rates) - min(rates))
        score = math.log1p(supports[-1]) * max(0.0, conditional - base_rate) * stability
        candidates.append({
            "raw": raw, "id": edge_id, "applicability": applicability,
            "requirement": requirement, "source_node": applicability[0]["left_column"],
            "target_node": requirement["left_column"], "score": score,
        })

    graph: dict[str, set[str]] = {}
    selected = []
    for candidate in sorted(candidates, key=lambda item: (-item["score"], item["id"])):
        source_node, target_node = candidate["source_node"], candidate["target_node"]
        if has_path(graph, source_node, target_node):
            audit.append({"candidate_id": candidate["id"], "decision": "exclude",
                          "reason": "transitive_shortcut", "score": candidate["score"]})
            continue
        if has_path(graph, target_node, source_node):
            audit.append({"candidate_id": candidate["id"], "decision": "exclude",
                          "reason": "would_create_cycle", "score": candidate["score"]})
            continue
        graph.setdefault(source_node, set()).add(target_node)
        raw = candidate["raw"]
        selected.append({
            "description": machine_description(candidate["applicability"], candidate["requirement"]),
            "applicability": candidate["applicability"], "requirements": [candidate["requirement"]],
            "exceptions": [], "relation_family": "distributional",
            "hardness_prior": raw.get("hardness_prior", "soft"),
            "evidence_basis": "empirical_signal",
            "rationale": "deterministic selection from discovery/counterexample/calibration statistics",
            "directionality_reason": "direction retained from train-only empirical implication mining",
            "plausible_counterexample": "captured by the calibrated non-zero exception rate",
            "confidence": 1.0 - float(raw["calibration"]["violation_rate"]),
            "candidate_id": "data_" + candidate["id"], "source": "empirical_mining",
            "discovery": raw["discovery"], "counterexample": raw["counterexample"],
            "calibration": raw["calibration"], "classification": raw.get("classification", "soft"),
            "provenance": ["empirical_mining", "deterministic_train_selector"],
            "usage_role": "actuate", "selection_score": candidate["score"],
        })
        audit.append({"candidate_id": candidate["id"], "decision": "select",
                      "score": candidate["score"], "rank": len(selected)})
        if len(selected) >= args.max_edges:
            break

    (args.output / "graph.json").write_text(json.dumps(selected, indent=2))
    (args.output / "selection_audit.json").write_text(json.dumps(audit, indent=2))
    metadata = {
        "method": "deterministic empirical DAG selector", "api_calls": 0,
        "llm_used": False, "heldout_fields_read": False,
        "source_candidates": len(source), "eligible_candidates": len(candidates),
        "selected_edges": len(selected), "selection_partitions": list(PARTITIONS),
        "filters": {"min_support": args.min_support, "min_lift": args.min_lift,
                    "max_rate": args.max_rate, "max_spread": args.max_spread},
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    for edge in selected:
        print(f"{edge['candidate_id']}: {edge['description']} score={edge['selection_score']:.4f}")


if __name__ == "__main__":
    main()
