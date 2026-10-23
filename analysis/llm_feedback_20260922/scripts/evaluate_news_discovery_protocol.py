#!/usr/bin/env python3
"""Audit privacy-safe News graph discovery against withheld public relations.

The five public relations are not exhaustive, so this script reports recall and
direction accuracy rather than calling every additional candidate a false
positive.  No source rows are emitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GOLD = {
    (" n_tokens_title", "<=", " n_tokens_content"),
    (" n_non_stop_unique_tokens", "<=", " n_non_stop_words"),
    (" num_self_hrefs", "<=", " num_hrefs"),
    (" self_reference_min_shares", "<=", " self_reference_max_shares"),
    (" self_reference_avg_sharess", "<=", " self_reference_max_shares"),
}
GOLD_ENDPOINTS = {frozenset((left, right)) for left, _, right in GOLD}


def normalized_column_atom(atom: dict) -> tuple[str, str, str] | None:
    if atom.get("right_kind") != "column":
        return None
    left, operator, right = atom.get("left_column"), atom.get("operator"), atom.get("right_column")
    if operator == ">=":
        return right, "<=", left
    if operator == ">":
        return right, "<", left
    return left, operator, right


def graph_metrics(path: Path, label: str) -> dict:
    graph = json.loads(path.read_text())
    exact, endpoint, reversed_direction = set(), set(), 0
    arities, conditional, multi_requirement = [], 0, 0
    for edge in graph:
        columns = set()
        atoms = list(edge.get("applicability", [])) + list(edge.get("requirements", []))
        atoms += [atom for group in edge.get("exceptions", []) for atom in group.get("all_of", [])]
        for atom in atoms:
            columns.add(atom.get("left_column"))
            if atom.get("right_kind") == "column":
                columns.add(atom.get("right_column"))
        columns.discard(None); columns.discard(""); arities.append(len(columns))
        conditional += int(bool(edge.get("applicability") or edge.get("exceptions")))
        multi_requirement += int(len(edge.get("requirements", [])) > 1)
        if edge.get("applicability") or edge.get("exceptions") or len(edge.get("requirements", [])) != 1:
            continue
        atom = normalized_column_atom(edge["requirements"][0])
        if atom is None:
            continue
        pair = frozenset((atom[0], atom[2])); endpoint.add(pair)
        if atom in GOLD:
            exact.add(atom)
        elif pair in GOLD_ENDPOINTS:
            reversed_direction += 1
    return {
        "graph": label,
        "retained_edges": len(graph),
        "hard": sum(edge.get("classification") == "hard" for edge in graph),
        "near_hard": sum(edge.get("classification") == "near_hard" for edge in graph),
        "soft": sum(edge.get("classification") == "soft" for edge in graph),
        "conditional_edges": conditional,
        "multi_requirement_edges": multi_requirement,
        "mean_distinct_columns": sum(arities) / len(arities) if arities else 0.0,
        "public_exact_recovered": len(exact),
        "public_exact_recall": len(exact) / len(GOLD),
        "public_endpoint_recovered": len(endpoint & GOLD_ENDPOINTS),
        "public_endpoint_recall": len(endpoint & GOLD_ENDPOINTS) / len(GOLD_ENDPOINTS),
        "wrong_direction_on_public_endpoint": reversed_direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    specifications = [
        ("strong_single_pass_unjudged_graph.json", "single_pass_plus_calibration"),
        ("strong_single_pass_graph.json", "single_pass_plus_semantic_judge"),
        ("llm_protocol_graph.json", "multi_role_counterexample_protocol"),
        ("empirical_graph.json", "statistics_only"),
        ("unified_graph.json", "unified_protocol"),
    ]
    rows = [graph_metrics(args.run / filename, label) for filename, label in specifications]
    signals_path = args.run / "value_aware_dependency_signals.json"
    signals = json.loads(signals_path.read_text()) if signals_path.exists() else []
    signal_pairs = {frozenset((row["current_feature"], row["conditioning_feature"])) for row in signals}
    signal_recall = len(signal_pairs & GOLD_ENDPOINTS) / len(GOLD_ENDPOINTS)
    table = pd.DataFrame(rows); table.to_csv(args.output / "discovery_metrics.csv", index=False)
    summary = {
        "withheld_public_relations": len(GOLD),
        "public_list_is_exhaustive": False,
        "value_aware_signals": len(signals),
        "value_aware_public_endpoint_recall": signal_recall,
        "privacy": "schema_and_aggregates_only; no raw real rows sent or emitted",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# News constraint-discovery audit",
        "",
        "The five ConstrainedDGM News inequalities were withheld from proposal prompts and used only here. "
        "Because that public list is not exhaustive, exact recall and direction are reported; extra candidates are not automatically labelled false positives.",
        "",
        "| Discovery method | Edges | Hard | Near-hard | Soft | Conditional | Mean columns | Public exact recall | Endpoint recall | Wrong direction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['graph']} | {row['retained_edges']} | {row['hard']} | {row['near_hard']} | {row['soft']} | "
            f"{row['conditional_edges']} | {row['mean_distinct_columns']:.2f} | {row['public_exact_recall']:.3f} | "
            f"{row['public_endpoint_recall']:.3f} | {row['wrong_direction_on_public_endpoint']} |"
        )
    lines += [
        "",
        f"SAGE-style value-aware retrieval produced **{len(signals)}** feature/value signals and covered "
        f"**{signal_recall:.3f}** of the withheld public endpoint pairs. These signals are dependencies, not constraints.",
        "",
        "`single_pass` is a privacy-safe LCATS-style proxy, not an official LCATS reproduction: official LCATS exposes 20 real rows to its discovery prompt, while this run exposes only schema and aggregates.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(table.to_string(index=False)); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
