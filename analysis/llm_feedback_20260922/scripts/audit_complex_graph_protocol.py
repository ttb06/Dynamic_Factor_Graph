#!/usr/bin/env python3
"""Audit structural complexity and dynamic-actuation readiness of a frozen graph."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def atoms(item: dict) -> list[dict]:
    result = list(item.get("applicability", [])) + list(item.get("requirements", []))
    result += [atom for group in item.get("exceptions", []) for atom in group.get("all_of", [])]
    return result


def columns(item: dict) -> set[str]:
    result = {atom["left_column"] for atom in atoms(item)}
    result |= {atom["right_column"] for atom in atoms(item) if atom.get("right_kind") == "column"}
    return result


def characterize(item: dict) -> dict:
    applicability = item.get("applicability", [])
    requirements = item.get("requirements", [])
    exceptions = item.get("exceptions", [])
    column_comparison = any(atom.get("right_kind") == "column" for atom in requirements)
    arity = len(columns(item))
    conjunction = len(applicability) >= 2
    multi_requirement = len(requirements) >= 2
    has_exception = bool(exceptions)
    contextual_comparison = column_comparison and bool(applicability)
    complex_edge = arity >= 3 or conjunction or multi_requirement or has_exception or contextual_comparison
    editable = {atom["left_column"] for atom in requirements}
    editable |= {atom["right_column"] for atom in requirements if atom.get("right_kind") == "column"}
    # Deactivating an antecedent is logically possible but may be semantically
    # undesirable; counting it exposes rather than resolves repair ambiguity.
    editable |= {atom["left_column"] for atom in applicability}
    return {
        "candidate_id": item.get("candidate_id"), "description": item.get("description"),
        "classification": item.get("classification"), "relation_family": item.get("relation_family"),
        "proposal_role": item.get("proposal_role"), "proposal_round": item.get("proposal_round"),
        "generalizability": item.get("generalizability"), "arity": arity,
        "applicability_atoms": len(applicability), "requirement_atoms": len(requirements),
        "exception_groups": len(exceptions), "column_comparison": column_comparison,
        "conjunction": conjunction, "multi_requirement": multi_requirement,
        "has_exception": has_exception, "contextual_comparison": contextual_comparison,
        "complex_edge": complex_edge, "plausible_action_endpoints": len(editable),
        "action_ambiguous": len(editable) >= 2,
        "calibration_support": item.get("calibration", {}).get("support", 0),
        "calibration_violation_rate": item.get("calibration", {}).get("violation_rate", np.nan),
        "heldout_violation_rate": item.get("heldout", {}).get("violation_rate", np.nan),
        "heldout_shift": item.get("heldout_shift", np.nan),
        "judge_confidence": item.get("judge_confidence", np.nan),
        "evidence_basis": item.get("evidence_basis"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lcats-edges", type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    graph = json.loads((args.run / "llm_protocol_graph.json").read_text())
    table = pd.DataFrame([characterize(item) for item in graph])
    if table.empty:
        table = pd.DataFrame(columns=list(characterize({}).keys()))
    table.to_csv(args.output / "complex_edges.csv", index=False)

    rejections = json.loads((args.run / "stage_rejections.json").read_text())
    rejected = []
    for stage, values in rejections.items():
        for item in values:
            if not isinstance(item, dict) or not item.get("requirements"): continue
            row = characterize(item); row["stage"] = stage; row["rejection"] = item.get("rejection", "")
            rejected.append(row)
    rejected_frame = pd.DataFrame(rejected)
    rejected_frame.to_csv(args.output / "complex_rejections.csv", index=False)

    retained = len(table); complex_n = int(table.complex_edge.sum()) if retained else 0
    summary = {
        "retained": retained, "complex_retained": complex_n,
        "complex_fraction": complex_n / retained if retained else 0.0,
        "arity_ge_3": int(table.arity.ge(3).sum()) if retained else 0,
        "conjunction": int(table.conjunction.sum()) if retained else 0,
        "multi_requirement": int(table.multi_requirement.sum()) if retained else 0,
        "with_exception": int(table.has_exception.sum()) if retained else 0,
        "contextual_column_comparison": int(table.contextual_comparison.sum()) if retained else 0,
        "action_ambiguous": int(table.action_ambiguous.sum()) if retained else 0,
        "non_seed_complex": int((table.complex_edge & table.proposal_role.ne("predefined_seed")).sum()) if retained else 0,
        "stable_complex_005": int((table.complex_edge & table.heldout_shift.le(.05)).sum()) if retained else 0,
    }
    if args.lcats_edges and args.lcats_edges.exists():
        lcats = pd.read_csv(args.lcats_edges)
        summary["lcats_constraints"] = len(lcats)
        summary["lcats_multi_column"] = int(lcats.arity.gt(1).sum())
        summary["lcats_dynamic_ready"] = int(lcats.dynamic_ready.astype(str).str.lower().eq("true").sum())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    rejection_counts = Counter(rejected_frame.loc[rejected_frame.complex_edge, "rejection"]) if not rejected_frame.empty else Counter()
    lines = [
        "# Complex typed-graph audit", "", f"Summary: `{summary}`", "",
        "Complex means at least one of: arity ≥ 3, conjunction, multiple requirements, explicit exception, or a contextual column comparison.", "",
        "| ID | Class | Family | Arity | Antecedent | Requirement | Exceptions | Actions | Shift | Description |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in table[table.complex_edge].sort_values(["heldout_shift", "arity"]).itertuples(index=False):
        lines.append(
            f"| {row.candidate_id} | {row.classification} | {row.relation_family} | {row.arity} | "
            f"{row.applicability_atoms} | {row.requirement_atoms} | {row.exception_groups} | "
            f"{row.plausible_action_endpoints} | {row.heldout_shift:.4f} | {row.description} |"
        )
    lines += ["", "## Complex-candidate rejection reasons", "", "| Reason | Count |", "| --- | ---: |"]
    for reason, count in rejection_counts.most_common(): lines.append(f"| {reason or 'critic/judge/calibration'} | {count} |")
    lines += ["", "The audit measures executable structural coverage, not semantic precision. Human/domain annotation is still required for the final graph."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
