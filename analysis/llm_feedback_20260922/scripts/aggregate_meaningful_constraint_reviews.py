#!/usr/bin/env python3
"""Aggregate semantic-review runs against an independent documented catalog."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    summaries, decisions, catalogs = [], [], {}
    for path in args.inputs:
        metadata = json.loads((path/"run_metadata.json").read_text())
        catalog = json.loads((path/"catalog.json").read_text())
        model = str(metadata["model"]); catalogs[model] = catalog
        documented = {item["candidate_id"] for item in catalog if item.get("documented_match")}
        predicted = {item["candidate_id"] for item in catalog
                     if item.get("semantic_review", {}).get("decision") == "likely_documented_invariant"}
        accepted = {item["candidate_id"] for item in catalog if item.get("usage_role") == "actuate"}
        reviewer_agreement = [item.get("semantic_review", {}).get("agreement") for item in catalog]
        summaries.append({
            "model": model, "candidates": len(catalog), "documented_reference": len(documented),
            "llm_likely_documented": len(predicted),
            "documented_recall": len(predicted & documented)/max(len(documented), 1),
            "documented_precision_lower_bound": len(predicted & documented)/max(len(predicted), 1),
            "reviewer_agreement": sum(value is True for value in reviewer_agreement)/max(len(reviewer_agreement), 1),
            "actuated": len(accepted), "non_documented_actuated": len(accepted-documented),
            "tokens": metadata.get("total_tokens", 0),
        })
        for item in catalog:
            review = item.get("semantic_review", {})
            decisions.append({
                "model": model, "candidate_id": item["candidate_id"],
                "description": item["description"], "family": item["relation_family"],
                "decision": review.get("decision"), "confidence": review.get("confidence"),
                "reviewer_agreement": review.get("agreement"),
                "documented": bool(item.get("documented_match")), "usage_role": item.get("usage_role"),
            })

    models = sorted(catalogs)
    cross = {}
    if len(models) >= 2:
        first = {item["candidate_id"]: item for item in catalogs[models[0]]}
        second = {item["candidate_id"]: item for item in catalogs[models[1]]}
        common = sorted(set(first) & set(second))
        matching = sum(
            first[key].get("semantic_review", {}).get("decision")
            == second[key].get("semantic_review", {}).get("decision") for key in common)
        cross = {"models": models[:2], "common_candidates": len(common),
                 "decision_agreement": matching/max(len(common), 1),
                 "actuation_graph_identical": {
                     key for key, item in first.items() if item.get("usage_role") == "actuate"
                 } == {
                     key for key, item in second.items() if item.get("usage_role") == "actuate"
                 }}

    summary = pd.DataFrame(summaries)
    detail = pd.DataFrame(decisions)
    summary.to_csv(args.output/"review_summary.csv", index=False)
    detail.to_csv(args.output/"candidate_decisions.csv", index=False)
    (args.output/"cross_model.json").write_text(json.dumps(cross, indent=2))
    columns = list(summary.columns)
    markdown = ["| " + " | ".join(columns) + " |",
                "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in summary.iterrows():
        markdown.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    lines = ["# Semantic constraint-review audit", "", *markdown, "",
             "The documented precision is a lower bound because the nine-edge reference catalog is not exhaustive. "
             "The safety property is exact: no edge without independent documentation may enter the actuation graph.", "",
             "## Cross-model", "", "```json", json.dumps(cross, indent=2), "```", "",
             "## Decision counts", ""]
    for model in models:
        counts = Counter(detail.loc[detail.model == model, "decision"].fillna("missing"))
        lines.extend([f"- `{model}`: {dict(counts)}"])
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(summary.to_string(index=False)); print(json.dumps(cross, indent=2))


if __name__ == "__main__":
    main()
