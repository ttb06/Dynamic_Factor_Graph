#!/usr/bin/env python3
"""Separate a broad verified graph from the smaller graph safe for actuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def atom_key(atom: dict) -> str:
    return json.dumps(atom, sort_keys=True)


def columns(item: dict) -> set[str]:
    atoms = item.get("applicability", []) + item.get("requirements", [])
    atoms += [atom for group in item.get("exceptions", []) for atom in group.get("all_of", [])]
    result = {atom["left_column"] for atom in atoms}
    result |= {atom["right_column"] for atom in atoms if atom.get("right_kind") == "column"}
    return result


def directionally_actionable(item: dict) -> tuple[bool, str]:
    """Do not fabricate upstream event counts to satisfy downstream aggregate duration."""
    app_columns = {atom["left_column"] for atom in item.get("applicability", [])}
    req_columns = {atom["left_column"] for atom in item.get("requirements", [])}
    for base in ["Administrative", "Informational", "ProductRelated"]:
        duration = f"{base}_Duration"
        if duration in app_columns and base in req_columns:
            return False, "reverse_measurement_action_would_fabricate_count"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-judge-confidence", type=float, default=.90)
    parser.add_argument("--max-heldout-shift", type=float, default=.05)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.graph.read_text())

    verifier, candidates, rejected = [], [], []
    for item in source:
        reason = None
        if item.get("classification") not in {"hard", "near_hard"}: reason = "soft_not_actuated"
        elif float(item.get("judge_confidence", 0)) < args.min_judge_confidence: reason = "low_semantic_confidence"
        elif float(item.get("heldout_shift", 1)) > args.max_heldout_shift: reason = "heldout_shift"
        elif (item.get("evidence_basis") == "empirical_signal"
              and item.get("classification") == "near_hard"
              and item.get("relation_family") in {"domain", "distributional"}):
            reason = "stable_association_not_actionable_semantics"
        if reason:
            rejected.append({"candidate_id": item.get("candidate_id"), "description": item.get("description"), "reason": reason})
            continue
        verifier.append(item)
        ok, reason = directionally_actionable(item)
        if ok: candidates.append(item)
        else: rejected.append({"candidate_id": item.get("candidate_id"), "description": item.get("description"), "reason": reason})

    # Remove threshold-specialized implications when a broader implication has
    # the same requirement and applicability column but greater support.
    by_signature = {}
    for item in candidates:
        signature = (
            tuple(sorted(atom_key(x) for x in item.get("requirements", []))),
            tuple(sorted(x["left_column"] for x in item.get("applicability", []))),
        )
        score = (float(item.get("discovery", {}).get("coverage", 0)),
                 float(item.get("judge_confidence", 0)))
        if signature not in by_signature or score > by_signature[signature][0]:
            if signature in by_signature:
                old = by_signature[signature][1]
                rejected.append({"candidate_id": old.get("candidate_id"), "description": old.get("description"),
                                 "reason": "subsumed_by_broader_edge"})
            by_signature[signature] = (score, item)
        else:
            rejected.append({"candidate_id": item.get("candidate_id"), "description": item.get("description"),
                             "reason": "subsumed_by_broader_edge"})
    candidates = [value[1] for value in by_signature.values()]

    # For duplicate unconditional column comparisons, prefer the explicitly
    # exception-aware edge, then semantic confidence.
    dedup = {}
    for item in candidates:
        req = tuple(sorted(atom_key(x) for x in item.get("requirements", [])))
        app = tuple(sorted(atom_key(x) for x in item.get("applicability", [])))
        signature = (req, app)
        score = (bool(item.get("exceptions")), float(item.get("judge_confidence", 0)))
        if signature not in dedup or score > dedup[signature][0]:
            if signature in dedup:
                old = dedup[signature][1]
                rejected.append({"candidate_id": old.get("candidate_id"), "description": old.get("description"),
                                 "reason": "duplicate_semantics"})
            dedup[signature] = (score, item)
        else:
            rejected.append({"candidate_id": item.get("candidate_id"), "description": item.get("description"),
                             "reason": "duplicate_semantics"})
    active = [value[1] for value in dedup.values()]
    active.sort(key=lambda x: ({"hard": 0, "near_hard": 1}[x["classification"]], -float(x.get("judge_confidence", 0))))

    (args.output/"verifier_graph.json").write_text(json.dumps(verifier, indent=2))
    (args.output/"active_graph.json").write_text(json.dumps(active, indent=2))
    (args.output/"rejections.json").write_text(json.dumps(rejected, indent=2))
    summary = {"source_edges": len(source), "verifier_edges": len(verifier), "active_edges": len(active),
               "hard": sum(x["classification"] == "hard" for x in active),
               "near_hard": sum(x["classification"] == "near_hard" for x in active),
               "column_comparisons": sum(any(a.get("right_kind") == "column" for a in x["requirements"]) for x in active),
               "rejected": len(rejected)}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Typed graph actuation curation", "", f"- Source edges: **{len(source)}**",
             f"- Verifier edges: **{len(verifier)}**", f"- Active edges: **{len(active)}**", "",
             "| ID | Class | Family | Description |", "| --- | --- | --- | --- |"]
    for item in active:
        lines.append(f"| {item.get('candidate_id')} | {item['classification']} | {item['relation_family']} | {item['description']} |")
    lines += ["", "Rejected edges remain available in `rejections.json`; curation does not modify the frozen source graph."]
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2)); print("\n".join(lines))


if __name__ == "__main__": main()
