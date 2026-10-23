#!/usr/bin/env python3
"""Curate mined constraints without promoting empirical regularities to truth."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-column-pair", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.candidates)
    for column in [
        "train_consequent_base_rate", "train_lift", "train_coverage", "test_coverage",
        "train_violation_rate", "test_violation_rate",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["eligible_empirical"] = frame["use_category"].isin(
        ["executable_baseline_candidate", "near_hard_dynamic_benchmark"]
    )
    frame["trivial_consequent"] = frame["train_consequent_base_rate"].ge(0.95).fillna(False)
    frame["weak_lift"] = frame["train_lift"].le(1.05).fillna(False)
    frame["nontrivial_implication"] = (
        frame["train_consequent_base_rate"].notna()
        & frame["train_consequent_base_rate"].lt(0.95)
        & frame["train_lift"].ge(1.20)
    )
    frame["mapping_candidate"] = frame["source"].eq("train_only_modal_mapping")
    frame["documented_candidate"] = frame["source"].eq("data_dictionary_derived_audit")
    frame["stable_heldout"] = frame["test_violation_rate"].le(0.01)
    frame["curation_pass"] = (
        frame["eligible_empirical"]
        & ~frame["sensitive_attribute"].astype(bool)
        & ~frame["target_related"].astype(bool)
        & frame["stable_heldout"]
        & (frame["nontrivial_implication"] | frame["mapping_candidate"] | frame["documented_candidate"])
    )
    frame["semantic_status"] = np.where(
        frame["source"].eq("data_dictionary_derived_audit"),
        "data_dictionary_derived_requires_independent_review",
        np.where(
            frame["source"].eq("pre_registered_interpretable_audit"),
            "pre_registered_but_still_requires_domain_source",
            "empirical_candidate_requires_external_semantic_validation",
        ),
    )
    frame.to_csv(args.output / "all_candidates_with_curation_flags.csv", index=False)

    curated = frame[frame["curation_pass"]].copy()
    curated["column_pair"] = (
        curated["antecedent_column"].fillna("<none>").astype(str)
        + " -> " + curated["consequent_column"].fillna("<none>").astype(str)
    )
    curated = curated.sort_values(
        ["dataset", "column_pair", "test_violation_rate", "train_lift", "train_applicable"],
        ascending=[True, True, True, False, False],
    )
    curated["pair_rank"] = curated.groupby(["dataset", "column_pair"]).cumcount() + 1
    shortlist = curated[curated["pair_rank"] <= args.max_per_column_pair].copy()
    shortlist.to_csv(args.output / "curated_empirical_shortlist.csv", index=False)

    summary_rows = []
    for dataset, block in frame.groupby("dataset"):
        eligible = block[block["eligible_empirical"]]
        summary_rows.append({
            "dataset": dataset,
            "all_candidates": len(block),
            "exact_or_near_hard": len(eligible),
            "trivial_base_ge_95pct": int(eligible["trivial_consequent"].sum()),
            "weak_lift_le_1_05": int(eligible["weak_lift"].sum()),
            "curation_pass_before_pair_dedup": int(block["curation_pass"].sum()),
            "documented_total": int(block["documented_candidate"].sum()),
            "documented_pass": int((block["documented_candidate"] & block["curation_pass"]).sum()),
            "empirical_pass": int((~block["documented_candidate"] & block["curation_pass"]).sum()),
            "curated_distinct_programs": int((shortlist["dataset"] == dataset).sum()),
            "distinct_column_pairs": int(shortlist.loc[shortlist["dataset"] == dataset, "column_pair"].nunique()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "curation_summary.csv", index=False)

    lines = [
        "# Constraint benchmark curation audit",
        "",
        "This audit is deliberately conservative. Passing the filters means a relation is a useful empirical stress candidate, not a semantic ground truth.",
        "",
        "| Dataset | All | Exact/near-hard | Trivial base ≥95% | Weak lift ≤1.05 | Documented pass | Empirical pass | Pair-deduplicated | Column pairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.all_candidates} | {row.exact_or_near_hard} | "
            f"{row.trivial_base_ge_95pct} | {row.weak_lift_le_1_05} | "
            f"{row.documented_pass} | {row.empirical_pass} | {row.curated_distinct_programs} | "
            f"{row.distinct_column_pairs} |"
        )
    lines.extend([
        "",
        "## Guardrails",
        "",
        "- Consequent base rate ≥95% is treated as a trivial-rule warning even when violations are zero.",
        "- Implications need lift ≥1.20; modal mappings are kept separately.",
        "- Sensitive/demographic and supervised-target relations are excluded.",
        "- At most two programs per ordered column pair are retained to prevent threshold variants from inflating benchmark size.",
        "- Every retained empirical relation still needs a data dictionary, domain source, or explicit labeling before it can be scored as semantic validity.",
    ])
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
