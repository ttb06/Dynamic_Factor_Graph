#!/usr/bin/env python3
"""Train-discovered, heldout-verified subgroup heterogeneity in constraint graphs.

This audit does not assert that empirical implications are semantic truths.  It
asks a narrower question: does the reliability of a candidate edge change
across an interpretable subgroup, and does that change replicate on heldout
data?  No source rows are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def subgroup_candidates(predicates, endpoint_columns: set[str], target: str | None,
                        max_per_column: int, max_total: int):
    by_column: dict[str, list] = {}
    for predicate in predicates:
        if predicate.column in endpoint_columns or predicate.column == target:
            continue
        coverage = float(predicate.train_mask.mean())
        test_support = int(predicate.test_mask.sum())
        if not 0.05 <= coverage <= 0.95 or test_support < 30:
            continue
        by_column.setdefault(predicate.column, []).append(predicate)
    kept = []
    for column, options in by_column.items():
        options.sort(key=lambda p: abs(float(p.train_mask.mean()) - 0.5))
        kept.extend(options[:max_per_column])
    kept.sort(key=lambda p: abs(float(p.train_mask.mean()) - 0.5))
    return kept[:max_total]


def contingency(antecedent: np.ndarray, consequent: np.ndarray, group: np.ndarray):
    active_group = antecedent & group
    active_other = antecedent & ~group
    vg = int((active_group & ~consequent).sum())
    sg = int((active_group & consequent).sum())
    vo = int((active_other & ~consequent).sum())
    so = int((active_other & consequent).sum())
    return vg, sg, vo, so


def rate(bad: int, good: int) -> float:
    return bad / (bad + good) if bad + good else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["adult", "default", "shoppers", "diabetes", "beijing", "news", "magic"])
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--min-edge-support", type=int, default=500)
    parser.add_argument("--min-cell-support", type=int, default=100)
    parser.add_argument("--min-test-cell-support", type=int, default=30)
    parser.add_argument("--min-train-gap", type=float, default=0.10)
    parser.add_argument("--min-test-gap", type=float, default=0.05)
    parser.add_argument("--min-train-interaction", type=float, default=0.05)
    parser.add_argument("--min-test-interaction", type=float, default=0.025)
    parser.add_argument("--max-relations", type=int, default=80)
    parser.add_argument("--max-groups", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo / "analysis/llm_feedback_20260922/scripts"))
    import cross_dataset_constraint_audit as audit

    catalog = pd.read_csv(args.candidate_file)
    rows: list[dict] = []
    dataset_rows: list[dict] = []
    schemas: dict[str, dict] = {}
    for dataset in args.datasets:
        train, test, categorical, numerical, target = audit.load_dataset(args.repo, dataset)
        predicates = audit.predicate_bank(train, test, categorical, numerical, args.min_support)
        lookup = {p.predicate_id: p for p in predicates}
        relations = catalog[
            catalog.dataset.eq(dataset)
            & catalog.source.eq("train_only_predicate_mining")
            & catalog.train_applicable.ge(args.min_edge_support)
            & catalog.train_violations.ge(20)
            & ~catalog.target_related.astype(bool)
        ].copy()
        # Prefer edges with enough uncertainty to expose meaningful subgroup
        # heterogeneity; exact mechanical identities are not the target here.
        relations["information"] = relations.train_applicable * relations.train_violation_rate * (1 - relations.train_violation_rate)
        relations = relations.sort_values(["information", "train_lift"], ascending=False).head(args.max_relations)
        evaluated = 0
        for relation in relations.itertuples(index=False):
            encoded = str(relation.constraint_id).split(":", 1)[1]
            if "->" not in encoded:
                continue
            antecedent_id, consequent_id = encoded.split("->", 1)
            antecedent, consequent = lookup.get(antecedent_id), lookup.get(consequent_id)
            if antecedent is None or consequent is None:
                continue
            groups = subgroup_candidates(
                predicates, {antecedent.column, consequent.column}, target,
                max_per_column=4, max_total=args.max_groups,
            )
            for group in groups:
                trg = contingency(antecedent.train_mask, consequent.train_mask, group.train_mask)
                teg = contingency(antecedent.test_mask, consequent.test_mask, group.test_mask)
                if min(trg[0] + trg[1], trg[2] + trg[3]) < args.min_cell_support:
                    continue
                if min(teg[0] + teg[1], teg[2] + teg[3]) < args.min_test_cell_support:
                    continue
                train_group_rate, train_other_rate = rate(trg[0], trg[1]), rate(trg[2], trg[3])
                test_group_rate, test_other_rate = rate(teg[0], teg[1]), rate(teg[2], teg[3])
                train_gap = train_group_rate - train_other_rate
                test_gap = test_group_rate - test_other_rate
                if abs(train_gap) < args.min_train_gap:
                    continue
                train_group_base = float(consequent.train_mask[group.train_mask].mean())
                train_other_base = float(consequent.train_mask[~group.train_mask].mean())
                test_group_base = float(consequent.test_mask[group.test_mask].mean())
                test_other_base = float(consequent.test_mask[~group.test_mask].mean())
                train_group_uplift = (1.0 - train_group_rate) - train_group_base
                train_other_uplift = (1.0 - train_other_rate) - train_other_base
                test_group_uplift = (1.0 - test_group_rate) - test_group_base
                test_other_uplift = (1.0 - test_other_rate) - test_other_base
                train_interaction = train_group_uplift - train_other_uplift
                test_interaction = test_group_uplift - test_other_uplift
                # A raw reliability gap is insufficient: G may simply predict
                # the consequent. Require the edge A->C to have different
                # uplift beyond each subgroup's own consequent base rate.
                if abs(train_interaction) < args.min_train_interaction:
                    continue
                if np.sign(train_interaction) != -np.sign(train_gap):
                    continue
                _, pvalue = fisher_exact([[trg[0], trg[1]], [trg[2], trg[3]]])
                sensitive = audit.is_sensitive([antecedent.column, consequent.column, group.column])
                candidate_id = "sg_" + hashlib.sha256(
                    f"{dataset}|{relation.constraint_id}|{group.predicate_id}".encode()
                ).hexdigest()[:16]
                rows.append({
                    "candidate_id": candidate_id,
                    "dataset": dataset,
                    "constraint_id": relation.constraint_id,
                    "constraint": relation.constraint,
                    "antecedent_column": antecedent.column,
                    "consequent_column": consequent.column,
                    "subgroup_id": group.predicate_id,
                    "subgroup": group.description,
                    "subgroup_column": group.column,
                    "train_group_support": trg[0] + trg[1],
                    "train_other_support": trg[2] + trg[3],
                    "train_group_violation": train_group_rate,
                    "train_other_violation": train_other_rate,
                    "train_gap": train_gap,
                    "train_group_consequent_base": train_group_base,
                    "train_other_consequent_base": train_other_base,
                    "train_group_edge_uplift": train_group_uplift,
                    "train_other_edge_uplift": train_other_uplift,
                    "train_interaction": train_interaction,
                    "test_group_support": teg[0] + teg[1],
                    "test_other_support": teg[2] + teg[3],
                    "test_group_violation": test_group_rate,
                    "test_other_violation": test_other_rate,
                    "test_gap": test_gap,
                    "test_group_consequent_base": test_group_base,
                    "test_other_consequent_base": test_other_base,
                    "test_group_edge_uplift": test_group_uplift,
                    "test_other_edge_uplift": test_other_uplift,
                    "test_interaction": test_interaction,
                    "same_direction": bool(
                        np.sign(train_gap) == np.sign(test_gap)
                        and np.sign(train_interaction) == np.sign(test_interaction)
                    ),
                    "heldout_effect": bool(
                        np.sign(train_gap) == np.sign(test_gap)
                        and abs(test_gap) >= args.min_test_gap
                        and np.sign(train_interaction) == np.sign(test_interaction)
                        and abs(test_interaction) >= args.min_test_interaction
                    ),
                    "fisher_p": pvalue,
                    "sensitive_or_demographic": sensitive,
                    "target_related": False,
                    "graph_interpretation": "edge_active_in_subgroup" if train_interaction > 0 else "exception_subgroup",
                    "train_only_discovery": True,
                })
                evaluated += 1
        dataset_rows.append({
            "dataset": dataset,
            "train_rows": len(train),
            "test_rows": len(test),
            "candidate_relations": len(relations),
            "reported_train_heterogeneity_pairs": evaluated,
        })
        schemas[dataset] = {
            "columns": [{"name": c, "type": "categorical" if c in categorical else "numeric"} for c in train.columns],
            "target": target,
        }

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no subgroup heterogeneity candidates passed train thresholds")
    frame["fdr_q"] = np.nan
    for dataset, indices in frame.groupby("dataset").groups.items():
        frame.loc[indices, "fdr_q"] = bh_adjust(frame.loc[indices, "fisher_p"].to_numpy())
    frame["train_significant"] = frame.fdr_q.le(0.05)
    frame["replicated"] = frame.train_significant & frame.heldout_effect
    frame.to_csv(args.output / "all_subgroup_edge_candidates.csv", index=False)
    replicated = frame[frame.replicated].copy().sort_values(
        ["sensitive_or_demographic", "dataset", "test_gap"], ascending=[True, True, False]
    )
    replicated.to_csv(args.output / "heldout_replicated_subgroup_edges.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(args.output / "dataset_summary.csv", index=False)
    (args.output / "schema_only.json").write_text(json.dumps(schemas, indent=2))

    # API payload contains schema and train aggregates only.  Heldout labels and
    # rates remain local for later evaluation.
    api_candidates = frame[frame.train_significant].copy()
    api_candidates = api_candidates.reindex(
        api_candidates.train_gap.abs().sort_values(ascending=False).index
    ).groupby("dataset").head(25)
    api_payload = []
    for dataset, part in api_candidates.groupby("dataset"):
        api_payload.append({
            "dataset": dataset,
            "schema": schemas[dataset],
            "privacy": "schema_and_train_aggregates_only",
            "candidate_edges": part[[
                "candidate_id", "constraint", "subgroup",
                "train_group_support", "train_other_support",
                "train_group_violation", "train_other_violation", "graph_interpretation",
                "train_group_edge_uplift", "train_other_edge_uplift", "train_interaction",
                "sensitive_or_demographic",
            ]].to_dict(orient="records"),
        })
    (args.output / "llm_train_only_payload.json").write_text(json.dumps(api_payload, indent=2))

    counts = frame.groupby("dataset").agg(
        train_candidates=("constraint_id", "size"),
        significant=("train_significant", "sum"),
        replicated=("replicated", "sum"),
        safe_replicated=("replicated", lambda x: int((x & ~frame.loc[x.index, "sensitive_or_demographic"]).sum())),
    ).reset_index()
    counts.to_csv(args.output / "replication_summary.csv", index=False)
    top = replicated[~replicated.sensitive_or_demographic].copy()
    top["abs_test_gap"] = top.test_gap.abs()
    top = top.sort_values("abs_test_gap", ascending=False).head(30)
    lines = [
        "# Subgroup-conditioned constraint-graph audit", "",
        "Subgroups and edge heterogeneity were selected using the training split. The heldout split was used only for replication. No source rows are written or sent to an API.", "",
        "## Replication summary", "",
        "| Dataset | Train candidates | FDR-significant | Heldout replicated | Non-sensitive replicated |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.train_candidates} | {row.significant} | {row.replicated} | {row.safe_replicated} |")
    lines += ["", "## Strongest non-sensitive replicated examples", "",
              "| Dataset | Edge | Subgroup | Train violation group/other | Test violation group/other | Interpretation |",
              "| --- | --- | --- | ---: | ---: | --- |"]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.constraint} | {row.subgroup} | "
            f"{row.train_group_violation:.3f}/{row.train_other_violation:.3f} | "
            f"{row.test_group_violation:.3f}/{row.test_other_violation:.3f} | {row.graph_interpretation} |"
        )
    lines += ["", "## Interpretation", "",
              "- `exception_subgroup`: the edge fails more often inside the subgroup; a global hard graph risks erasing valid exceptions.",
              "- `edge_active_in_subgroup`: the edge is more reliable inside the subgroup; a global graph can miss useful local structure.",
              "- Sensitive or target-related relations are never candidates for enforcement; they remain audit-only.",
              "- Replication establishes subgroup heterogeneity, not semantic validity. LLM selection must still pass typed execution and independent verification."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    metadata = {
        "status": "executed", "datasets": args.datasets,
        "test_used_for_discovery": False, "raw_rows_written": False,
        "api_payload_contains_test_metrics": False,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(counts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
