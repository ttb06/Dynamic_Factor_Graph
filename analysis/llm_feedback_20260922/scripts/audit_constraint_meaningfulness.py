#!/usr/bin/env python3
"""Counterfactual and semantic audit for LLM-proposed typed constraints.

This is deliberately stricter than the protocol's held-out frequency check.
It asks whether every atom changes the executable relation, whether two edges
detect effectively the same rows, and whether an edge has a semantic rather
than merely empirical justification.  The generated review is an aid for a
human decision; it never promotes an edge automatically into the sampler.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from knowledge_graph_protocol import predicate_mask


def edge_masks(frame: pd.DataFrame, edge: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    antecedent = np.ones(len(frame), dtype=bool)
    for atom in edge.get("applicability", []):
        antecedent &= predicate_mask(frame, atom)
    exception = np.zeros(len(frame), dtype=bool)
    for group in edge.get("exceptions", []):
        group_mask = np.ones(len(frame), dtype=bool)
        for atom in group.get("all_of", []):
            group_mask &= predicate_mask(frame, atom)
        exception |= group_mask
    active = antecedent & ~exception
    satisfied = np.ones(len(frame), dtype=bool)
    for atom in edge.get("requirements", []):
        satisfied &= predicate_mask(frame, atom)
    return antecedent, active, satisfied, active & ~satisfied


def atom_text(atom: dict) -> str:
    right = atom.get("right_column") if atom.get("right_kind") == "column" else repr(atom.get("right_value"))
    tolerance = float(atom.get("tolerance", 0) or 0)
    suffix = f" (tol={tolerance:g})" if tolerance else ""
    return f"{atom.get('left_column')} {atom.get('operator')} {right}{suffix}"


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else 1.0


def change_stats(original: np.ndarray, ablated: np.ndarray) -> tuple[int, float]:
    changed = int(np.logical_xor(original, ablated).sum())
    return changed, changed / len(original) if len(original) else math.nan


def _reference_value(row: pd.Series, atom: dict):
    if atom.get("right_kind") == "column":
        return row[atom["right_column"]]
    return atom.get("right_value")


def _numeric_delta(value, tolerance: float) -> float:
    try:
        numeric = float(value)
        return max(1e-3, abs(numeric) * .05, tolerance * 2 + 1e-5)
    except (TypeError, ValueError):
        return 1.0


def mutate_atom(row: pd.Series, atom: dict, truth: bool) -> pd.Series:
    """Set the left endpoint so one typed predicate is true or false.

    This creates an executor-level counterexample only.  It does not assert
    that the mutated row is semantically or distributionally plausible.
    """
    changed = row.copy()
    column = atom["left_column"]
    operator = atom["operator"]
    reference = _reference_value(changed, atom)
    tolerance = float(atom.get("tolerance", 0) or 0)
    numeric = pd.api.types.is_number(reference) and not pd.isna(reference)
    delta = _numeric_delta(reference, tolerance)
    if operator == "==":
        changed[column] = reference if truth else (float(reference) + delta if numeric else "__counterfactual__")
    elif operator == "!=":
        changed[column] = (float(reference) + delta if numeric else "__counterfactual__") if truth else reference
    elif operator == "<=":
        changed[column] = float(reference) if truth else float(reference) + tolerance + delta
    elif operator == "<":
        changed[column] = float(reference) - tolerance - delta if truth else float(reference) + tolerance
    elif operator == ">=":
        changed[column] = float(reference) if truth else float(reference) - tolerance - delta
    elif operator == ">":
        changed[column] = float(reference) + tolerance + delta if truth else float(reference) - tolerance
    return changed


def edge_violation_on_row(edge: dict, row: pd.Series) -> tuple[bool, bool]:
    frame = pd.DataFrame([row])
    _, active, _, violation = edge_masks(frame, edge)
    return bool(active[0]), bool(violation[0])


def satisfied_active_row(edge: dict, frame: pd.DataFrame) -> pd.Series | None:
    """Construct a local row satisfying the executable edge when possible."""
    _, active, satisfied, _ = edge_masks(frame, edge)
    candidates = np.flatnonzero(active & satisfied)
    if len(candidates):
        return frame.iloc[int(candidates[0])].copy()
    candidates = np.flatnonzero(active)
    if not len(candidates):
        return None
    row = frame.iloc[int(candidates[0])].copy()
    for atom in edge.get("requirements", []):
        row = mutate_atom(row, atom, True)
    return row


def requirement_counterfactual(edge: dict, frame: pd.DataFrame, index: int) -> tuple[bool, bool]:
    row = satisfied_active_row(edge, frame)
    if row is None:
        return False, False
    before_active, before_violation = edge_violation_on_row(edge, row)
    changed = mutate_atom(row, edge["requirements"][index], False)
    after_active, after_violation = edge_violation_on_row(edge, changed)
    return True, before_active and not before_violation and after_active and after_violation


def applicability_counterfactual(edge: dict, frame: pd.DataFrame, index: int) -> tuple[bool, bool]:
    row = satisfied_active_row(edge, frame)
    if row is None or not edge.get("requirements"):
        return False, False
    # Make a clean active violation, then turn off only the selected gate.
    row = mutate_atom(row, edge["requirements"][0], False)
    before_active, before_violation = edge_violation_on_row(edge, row)
    changed = mutate_atom(row, edge["applicability"][index], False)
    after_active, after_violation = edge_violation_on_row(edge, changed)
    return True, before_active and before_violation and not after_active and not after_violation


def exception_counterfactual(edge: dict, frame: pd.DataFrame, group_index: int, atom_index: int) -> tuple[bool, bool]:
    row = satisfied_active_row(edge, frame)
    if row is None or not edge.get("requirements"):
        return False, False
    row = mutate_atom(row, edge["requirements"][0], False)
    active_before, violation_before = edge_violation_on_row(edge, row)
    group = edge["exceptions"][group_index].get("all_of", [])
    exempted = row.copy()
    for atom in group:
        exempted = mutate_atom(exempted, atom, True)
    active_exempted, violation_exempted = edge_violation_on_row(edge, exempted)
    atom_removed = mutate_atom(exempted, group[atom_index], False)
    active_removed, violation_removed = edge_violation_on_row(edge, atom_removed)
    effective = (active_before and violation_before and not active_exempted and not violation_exempted
                 and active_removed and violation_removed)
    return True, effective


def structural_tokens(edge: dict, section: str) -> set[str]:
    if section == "exceptions":
        atoms = [atom for group in edge.get("exceptions", []) for atom in group.get("all_of", [])]
    else:
        atoms = edge.get(section, [])
    tokens = set()
    for atom in atoms:
        tolerance = float(atom.get("tolerance", 0) or 0)
        # Treat numerical tolerances below 1e-5 as the same executable atom;
        # otherwise duplicate proposals with 0 versus 1e-6 evade the audit.
        tolerance = 0.0 if tolerance <= 1e-5 else round(tolerance, 8)
        right = atom.get("right_column") if atom.get("right_kind") == "column" else atom.get("right_value")
        tokens.add(json.dumps([atom.get("left_column"), atom.get("operator"), atom.get("right_kind"),
                               right, tolerance], sort_keys=True))
    return tokens


def set_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def audit_edge(edge: dict, frame: pd.DataFrame, heldout: pd.DataFrame) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    antecedent, active, satisfied, violations = edge_masks(frame, edge)
    _, held_active, _, held_violations = edge_masks(heldout, edge)
    atom_rows: list[dict] = []
    flags: list[str] = []
    minimum_material_rows = max(10, int(math.ceil(.005 * len(frame))))

    for index, atom in enumerate(edge.get("applicability", [])):
        ablated = np.ones(len(frame), dtype=bool)
        for other_index, other in enumerate(edge.get("applicability", [])):
            if other_index != index:
                ablated &= predicate_mask(frame, other)
        exception = np.zeros(len(frame), dtype=bool)
        for group in edge.get("exceptions", []):
            group_mask = np.ones(len(frame), dtype=bool)
            for exception_atom in group.get("all_of", []):
                group_mask &= predicate_mask(frame, exception_atom)
            exception |= group_mask
        ablated_active = ablated & ~exception
        changed, fraction = change_stats(active, ablated_active)
        testable, effective = applicability_counterfactual(edge, frame, index)
        redundant = testable and not effective
        if redundant:
            flags.append(f"redundant_applicability_{index}")
        atom_rows.append({"candidate_id": edge.get("candidate_id"), "section": "applicability",
                          "atom_index": index, "atom": atom_text(atom), "changed_rows": changed,
                          "changed_fraction": fraction, "observationally_redundant": changed < minimum_material_rows,
                          "synthetic_testable": testable, "synthetic_effective": effective,
                          "counterfactually_redundant": redundant})

    requirements = edge.get("requirements", [])
    requirement_masks = [predicate_mask(frame, atom) for atom in requirements]
    for index, atom in enumerate(requirements):
        others = np.ones(len(frame), dtype=bool)
        for other_index, mask in enumerate(requirement_masks):
            if other_index != index:
                others &= mask
        unique_failures = active & others & ~requirement_masks[index]
        changed = int(unique_failures.sum())
        fraction = changed / len(frame) if len(frame) else math.nan
        testable, effective = requirement_counterfactual(edge, frame, index)
        # Natural data cannot establish redundancy for a true invariant: all
        # valid rows may satisfy every atom.  Use a minimally mutated row to
        # test whether the executor changes state when this atom alone fails.
        redundant = testable and not effective
        if redundant:
            flags.append(f"redundant_requirement_{index}")
        atom_rows.append({"candidate_id": edge.get("candidate_id"), "section": "requirement",
                          "atom_index": index, "atom": atom_text(atom), "changed_rows": changed,
                          "changed_fraction": fraction,
                          "observationally_redundant": changed < max(3, int(math.ceil(.001 * len(frame)))),
                          "synthetic_testable": testable, "synthetic_effective": effective,
                          "counterfactually_redundant": redundant})

    for group_index, group in enumerate(edge.get("exceptions", [])):
        atoms = group.get("all_of", [])
        group_mask = np.ones(len(frame), dtype=bool)
        for atom in atoms:
            group_mask &= predicate_mask(frame, atom)
        exempted = antecedent & group_mask
        if int(exempted.sum()) < max(3, int(math.ceil(.001 * len(frame)))):
            flags.append(f"vacuous_exception_group_{group_index}")
        for index, atom in enumerate(atoms):
            ablated = np.ones(len(frame), dtype=bool)
            for other_index, other in enumerate(atoms):
                if other_index != index:
                    ablated &= predicate_mask(frame, other)
            changed = int((antecedent & np.logical_xor(group_mask, ablated)).sum())
            fraction = changed / len(frame) if len(frame) else math.nan
            testable, effective = exception_counterfactual(edge, frame, group_index, index)
            redundant = testable and not effective
            if redundant:
                flags.append(f"redundant_exception_{group_index}_{index}")
            atom_rows.append({"candidate_id": edge.get("candidate_id"), "section": f"exception_{group_index}",
                              "atom_index": index, "atom": atom_text(atom), "changed_rows": changed,
                              "changed_fraction": fraction, "observationally_redundant": changed < minimum_material_rows,
                              "synthetic_testable": testable, "synthetic_effective": effective,
                              "counterfactually_redundant": redundant})

    family = str(edge.get("relation_family", ""))
    evidence = str(edge.get("evidence_basis", ""))
    generalizability = str(edge.get("generalizability", ""))
    base_rate = float(edge.get("calibration", {}).get("base_rate", np.nan))
    coverage = float(edge.get("calibration", {}).get("coverage", np.nan))
    support = int(edge.get("calibration", {}).get("support", int(active.sum())))
    violation_rate = float(edge.get("calibration", {}).get("violation_rate", np.nan))
    heldout_rate = float(edge.get("heldout", {}).get("violation_rate", np.nan))
    if family in {"distributional"}:
        flags.append(f"weak_semantic_family_{family}")
    if evidence == "empirical_signal" or generalizability in {"dataset_empirical", "not_generalizable"}:
        flags.append("empirical_or_nongeneralizable")
    if np.isfinite(base_rate) and base_rate >= .98:
        flags.append("near_constant_consequent")
    if support < 60 or (np.isfinite(coverage) and coverage < .01):
        flags.append("low_support")
    judge_issues = [str(x) for x in edge.get("judge_issues", []) if str(x) != "none"]
    flags.extend(f"judge_{issue}" for issue in judge_issues)

    critical_prefixes = ("redundant_", "vacuous_", "weak_semantic_", "judge_tautology", "judge_spurious",
                         "judge_wrong", "judge_normative", "judge_dataset_specific")
    critical = any(flag.startswith(critical_prefixes) for flag in flags)
    classification = str(edge.get("classification", edge.get("hardness_prior", "soft")))
    if critical:
        recommendation = "reject_or_revise"
    elif classification == "soft" or (np.isfinite(violation_rate) and violation_rate > .05):
        recommendation = "score_only"
    elif flags:
        recommendation = "human_review"
    else:
        recommendation = "actuation_candidate"

    row = {
        "candidate_id": edge.get("candidate_id"), "description": edge.get("description"),
        "relation_family": family, "classification": classification, "evidence_basis": evidence,
        "generalizability": generalizability, "judge_confidence": edge.get("judge_confidence", np.nan),
        "applicability": " AND ".join(atom_text(x) for x in edge.get("applicability", [])) or "always",
        "requirements": " AND ".join(atom_text(x) for x in requirements),
        "exceptions": " OR ".join("(" + " AND ".join(atom_text(x) for x in g.get("all_of", [])) + ")"
                                  for g in edge.get("exceptions", [])) or "none",
        "train_active": int(active.sum()), "train_violations": int(violations.sum()),
        "train_violation_rate": float(violations.sum() / active.sum()) if active.any() else np.nan,
        "heldout_active_recomputed": int(held_active.sum()), "heldout_violations_recomputed": int(held_violations.sum()),
        "heldout_violation_rate_recomputed": float(held_violations.sum() / held_active.sum()) if held_active.any() else np.nan,
        "calibration_base_rate": base_rate, "calibration_support": support,
        "calibration_violation_rate": violation_rate, "reported_heldout_violation_rate": heldout_rate,
        "semantic_flags": ";".join(dict.fromkeys(flags)), "recommendation": recommendation,
        "rationale": edge.get("rationale", ""), "judge_rationale": edge.get("judge_rationale", ""),
        "plausible_counterexample": edge.get("plausible_counterexample", ""),
    }
    return row, atom_rows, active, violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=20261521)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo / "analysis/llm_feedback_20260922/scripts"))
    import cross_dataset_constraint_audit as dataset_audit

    train, heldout, _, _, _ = dataset_audit.load_dataset(args.repo, args.dataset)
    graph = json.loads((args.run / "llm_protocol_graph.json").read_text())
    rows: list[dict] = []
    atom_rows: list[dict] = []
    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    edge_by_id: dict[str, dict] = {}
    for index, edge in enumerate(graph):
        edge = {**edge, "candidate_id": edge.get("candidate_id", f"edge_{index:03d}")}
        row, atoms, active, violations = audit_edge(edge, train, heldout)
        rows.append(row); atom_rows.extend(atoms); masks[str(edge["candidate_id"])] = (active, violations)
        edge_by_id[str(edge["candidate_id"])] = edge

    pairs = []
    duplicate_with: dict[str, list[str]] = {}
    identifiers = list(masks)
    for left_index, left in enumerate(identifiers):
        for right in identifiers[left_index + 1:]:
            active_j = jaccard(masks[left][0], masks[right][0])
            violation_union = int((masks[left][1] | masks[right][1]).sum())
            violation_j = jaccard(masks[left][1], masks[right][1]) if violation_union else math.nan
            requirement_j = set_jaccard(structural_tokens(edge_by_id[left], "requirements"),
                                        structural_tokens(edge_by_id[right], "requirements"))
            applicability_j = set_jaccard(structural_tokens(edge_by_id[left], "applicability"),
                                          structural_tokens(edge_by_id[right], "applicability"))
            exception_j = set_jaccard(structural_tokens(edge_by_id[left], "exceptions"),
                                      structural_tokens(edge_by_id[right], "exceptions"))
            near_duplicate = requirement_j >= .95 and applicability_j >= .95 and exception_j >= .95
            if near_duplicate or (violation_union and violation_j >= .95):
                pairs.append({"left": left, "right": right, "active_jaccard": active_j,
                              "violation_jaccard": violation_j, "requirement_jaccard": requirement_j,
                              "applicability_jaccard": applicability_j, "exception_jaccard": exception_j,
                              "near_duplicate": near_duplicate})
            if near_duplicate:
                duplicate_with.setdefault(right, []).append(left)

    for row in rows:
        duplicates = duplicate_with.get(str(row["candidate_id"]), [])
        row["duplicate_with"] = ";".join(duplicates)
        if duplicates:
            row["semantic_flags"] = ";".join(x for x in [row["semantic_flags"], "near_duplicate_edge"] if x)
            row["recommendation"] = "reject_or_revise"

    table = pd.DataFrame(rows)
    atoms = pd.DataFrame(atom_rows)
    pair_table = pd.DataFrame(pairs, columns=["left", "right", "active_jaccard", "violation_jaccard",
                                                  "requirement_jaccard", "applicability_jaccard",
                                                  "exception_jaccard", "near_duplicate"])
    table.to_csv(args.output / "constraint_meaningfulness.csv", index=False)
    atoms.to_csv(args.output / "atom_counterfactual_audit.csv", index=False)
    pair_table.to_csv(args.output / "edge_overlap_audit.csv", index=False)
    selected_ids = set(table.loc[table.recommendation.isin(["actuation_candidate", "human_review"]), "candidate_id"].astype(str))
    selected = [edge for edge in graph if str(edge.get("candidate_id")) in selected_ids]
    (args.output / "potentially_meaningful.json").write_text(json.dumps(selected, indent=2))

    counts = table.recommendation.value_counts().to_dict() if len(table) else {}
    summary = {"dataset": args.dataset, "edges": len(table), "recommendations": counts,
               "counterfactually_redundant_atoms": int(atoms.counterfactually_redundant.sum()) if len(atoms) else 0,
               "near_duplicate_pairs": int(pair_table.near_duplicate.sum()) if len(pair_table) else 0,
               "automatic_promotion": False, "human_review_required": True}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Semantic and counterfactual constraint review", "", f"Summary: `{summary}`", "",
             "No edge in this file is automatically authorized for guidance. The table is a human review queue.", "",
             "| ID | Recommendation | Type | Train support/viol. | Heldout support/viol. | Flags | Relation |",
             "| --- | --- | --- | ---: | ---: | --- | --- |"]
    if len(table):
        order = {"actuation_candidate": 0, "human_review": 1, "score_only": 2, "reject_or_revise": 3}
        for row in table.assign(_order=table.recommendation.map(order)).sort_values(["_order", "candidate_id"]).itertuples(index=False):
            lines.append(f"| {row.candidate_id} | {row.recommendation} | {row.classification}/{row.relation_family} | "
                         f"{row.train_active}/{row.train_violations} | {row.heldout_active_recomputed}/{row.heldout_violations_recomputed} | "
                         f"{row.semantic_flags or 'none'} | {row.description} |")
        lines += ["", "## Full executable definitions", ""]
        for row in table.itertuples(index=False):
            lines += [f"### {row.candidate_id}: {row.description}", "",
                      f"- Applies when: `{row.applicability}`", f"- Requires: `{row.requirements}`",
                      f"- Exceptions: `{row.exceptions}`", f"- Rationale: {row.rationale}",
                      f"- Independent judge: {row.judge_rationale}",
                      f"- Plausible counterexample: {row.plausible_counterexample}", ""]
    (args.output / "REVIEW.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
