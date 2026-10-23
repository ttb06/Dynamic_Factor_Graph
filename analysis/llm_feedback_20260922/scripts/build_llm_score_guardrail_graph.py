#!/usr/bin/env python3
"""Merge independently verified public actuators with LLM soft guardrails."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def atom_signature(atom: dict) -> tuple:
    right = atom.get("right_column") if atom.get("right_kind") == "column" else atom.get("right_value")
    return (atom.get("left_column"), atom.get("operator"), atom.get("right_kind"), str(right))


def edge_signature(edge: dict) -> tuple:
    applicability = tuple(sorted(atom_signature(atom) for atom in edge.get("applicability", [])))
    requirements = tuple(sorted(atom_signature(atom) for atom in edge.get("requirements", [])))
    exceptions = tuple(sorted(tuple(sorted(atom_signature(atom) for atom in group.get("all_of", [])))
                              for group in edge.get("exceptions", [])))
    return applicability, requirements, exceptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-graph", type=Path, required=True)
    parser.add_argument("--llm-run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--public-schedule", type=Path, required=True)
    parser.add_argument("--soft-cutoff", type=float, default=.50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    public = json.loads(args.public_graph.read_text())
    llm = json.loads((args.llm_run / "llm_protocol_graph.json").read_text())
    review = pd.read_csv(args.audit / "constraint_meaningfulness.csv")
    allowed = set(review.loc[
        review.recommendation.isin(["actuation_candidate", "human_review", "score_only"]),
        "candidate_id",
    ].astype(str))
    score_edges = []
    public_signatures = {edge_signature(edge) for edge in public}
    skipped_public_duplicates = 0
    for ordinal, edge in enumerate(llm):
        if str(edge.get("candidate_id")) not in allowed:
            continue
        if edge_signature(edge) in public_signatures:
            skipped_public_duplicates += 1
            continue
        copied = dict(edge)
        copied["original_candidate_id"] = copied.get("candidate_id")
        copied["candidate_id"] = f"news_llm_score_{ordinal:03d}"
        copied["usage_role"] = "score_only"
        copied["external_authority"] = False
        copied["source"] = "privacy_safe_llm_protocol_then_empirical_and_semantic_audit"
        score_edges.append(copied)

    graph = public + score_edges
    (args.output / "public_plus_llm_score_graph.json").write_text(json.dumps(graph, indent=2))
    (args.output / "public_only_graph.json").write_text(json.dumps(public, indent=2))
    schedule = pd.read_csv(args.public_schedule)
    extra = []
    for edge in score_edges:
        calibration = edge.get("calibration", {})
        heldout = edge.get("heldout", {})
        extra.append({
            "candidate_id": edge["candidate_id"], "description": edge.get("description", ""),
            "classification": edge.get("classification", "soft"), "cutoff": args.soft_cutoff,
            "family": edge.get("relation_family", "unknown"), "source": edge.get("source", ""),
            "usage_role": "score_only", "readable": True, "violation_evaluable": False,
            "final_support": heldout.get("support", 0), "final_violations": heldout.get("violations", 0),
            "synthetic_violation_rate": float("nan"),
            "heldout_violation_rate": heldout.get("violation_rate", float("nan")),
            "calibration_violation_rate": calibration.get("violation_rate", float("nan")),
        })
    merged_schedule = pd.concat([schedule, pd.DataFrame(extra)], ignore_index=True)
    merged_schedule.to_csv(args.output / "public_plus_llm_schedule.csv", index=False)
    schedule.to_csv(args.output / "public_only_schedule.csv", index=False)
    metadata = {
        "public_actuators": len(public), "llm_score_guardrails": len(score_edges),
        "skipped_exact_public_duplicates": skipped_public_duplicates,
        "llm_edges_never_actuate": True, "soft_cutoff": args.soft_cutoff,
        "selection": "semantic judge plus counterfactual meaningfulness audit",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
