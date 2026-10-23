#!/usr/bin/env python3
"""Build and audit conservative LCATS-style unary domain constraints.

LCATS Table-1 includes ``domain_bound`` and ``is_integer`` primitives.  The
main project graph previously contained only inter-column/functional factors.
This module adds schema-level unary constraints, verifies every candidate on
the train split, and emits the same typed graph format consumed by the dynamic
trajectory sampler.  No held-out row is used to create or retain an edge.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


NEWS_INTEGER = [
    " timedelta", " n_tokens_title", " n_tokens_content", " num_hrefs",
    " num_self_hrefs", " num_imgs", " num_videos", " num_keywords",
]

NEWS_NONNEGATIVE = NEWS_INTEGER + [
    " average_token_length", " self_reference_min_shares",
    " self_reference_max_shares", " self_reference_avg_sharess",
]

NEWS_UNIT_INTERVAL = [
    " LDA_00", " LDA_01", " LDA_02", " LDA_03", " LDA_04",
    " global_subjectivity", " global_rate_positive_words",
    " global_rate_negative_words", " rate_positive_words",
    " rate_negative_words", " avg_positive_polarity",
    " min_positive_polarity", " max_positive_polarity",
    " title_subjectivity", " abs_title_subjectivity",
    " abs_title_sentiment_polarity",
]

NEWS_SIGNED_UNIT = [
    " global_sentiment_polarity", " avg_negative_polarity",
    " min_negative_polarity", " max_negative_polarity",
    " title_sentiment_polarity",
]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def base_edge(candidate_id: str, family: str, description: str,
              columns: list[str]) -> dict:
    return {
        "candidate_id": candidate_id,
        "description": description,
        "columns": columns,
        "applicability": [],
        "exceptions": [],
        "relation_family": family,
        "hardness_prior": "hard",
        "classification": "hard",
        "usage_role": "actuate",
        "evidence_basis": "schema_semantics_and_train_verification",
        "source": "lcats_table1_domain_protocol",
        "provenance": [
            "schema_semantics", "lcats_domain_primitive", "train_only_verification"
        ],
        "confidence": 1.0,
        "guidance_weight": 1.0,
    }


def bound_edge(column: str, operator: str, value: float) -> dict:
    side = "lower" if operator == ">=" else "upper"
    value_tag = ("neg" if value < 0 else "") + slug(str(abs(value)))
    edge = base_edge(
        f"domain_{side}_{slug(column)}_{value_tag}",
        "domain_bound",
        f"Schema domain: {column.strip()} {operator} {value:g}", [column])
    edge["requirements"] = [{
        "left_column": column,
        "operator": operator,
        "right_kind": "literal",
        "right_column": "",
        "right_value": float(value),
        "tolerance": 1e-7,
    }]
    return edge


def integer_edge(column: str) -> dict:
    edge = base_edge(
        f"domain_integer_{slug(column)}", "integer_domain",
        f"Schema domain: {column.strip()} is integer-valued", [column])
    edge["requirements"] = []
    edge["program"] = {
        "kind": "integer", "column": column, "tolerance": 1e-6,
    }
    return edge


def news_candidates(families: set[str]) -> list[dict]:
    edges: list[dict] = []
    if "integer" in families:
        edges.extend(integer_edge(column) for column in NEWS_INTEGER)
    if "bounds" in families:
        edges.extend(bound_edge(column, ">=", 0.0) for column in NEWS_NONNEGATIVE)
        for column in NEWS_UNIT_INTERVAL:
            edges.append(bound_edge(column, ">=", 0.0))
            edges.append(bound_edge(column, "<=", 1.0))
        for column in NEWS_SIGNED_UNIT:
            edges.append(bound_edge(column, ">=", -1.0))
            edges.append(bound_edge(column, "<=", 1.0))
    return edges


def edge_violation(frame: pd.DataFrame, edge: dict) -> np.ndarray:
    if edge.get("program", {}).get("kind") == "integer":
        column = edge["program"]["column"]
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        return (~np.isfinite(values)) | (np.abs(values - np.rint(values)) > 1e-5)
    atom = edge["requirements"][0]
    values = pd.to_numeric(frame[atom["left_column"]], errors="coerce").to_numpy(float)
    target = float(atom["right_value"])
    if atom["operator"] == ">=":
        return (~np.isfinite(values)) | (values < target - 1e-7)
    return (~np.isfinite(values)) | (values > target + 1e-7)


def verified_candidates(dataset: str, train: pd.DataFrame,
                        families: set[str]) -> tuple[list[dict], list[dict]]:
    if dataset != "news":
        raise ValueError("the conservative semantic profile is currently defined for news")
    kept, rejected = [], []
    for edge in news_candidates(families):
        missing = [column for column in edge["columns"] if column not in train]
        if missing:
            rejected.append({**edge, "rejection": f"missing columns: {missing}"})
            continue
        violation = edge_violation(train, edge)
        record = {
            "support": int(len(train)),
            "violations": int(violation.sum()),
            "violation_rate": float(violation.mean()),
        }
        edge["discovery"] = record
        # Existing graph_metrics compares generated violations against the
        # frozen calibration rate.  Domain candidates are selected and frozen
        # on train, so the verified train rate is their calibration reference.
        edge["calibration"] = dict(record)
        if not violation.any():
            kept.append(edge)
        else:
            rejected.append({**edge, "rejection": "nonzero train violation"})
    return kept, rejected


def build(args: argparse.Namespace) -> None:
    train = pd.read_csv(args.train)
    base = json.loads(args.base_graph.read_text()) if args.base_graph else []
    if not isinstance(base, list):
        base = base.get("edges", base.get("constraints", []))
    families = {value.strip() for value in args.families.split(",") if value.strip()}
    unknown = families - {"integer", "bounds"}
    if unknown:
        raise ValueError(f"unknown domain families: {sorted(unknown)}")
    kept, rejected = verified_candidates(args.dataset, train, families)
    output = base + kept
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    summary = {
        "dataset": args.dataset,
        "train": str(args.train),
        "base_edges": len(base),
        "domain_candidates": len(kept) + len(rejected),
        "domain_retained": len(kept),
        "domain_rejected": len(rejected),
        "families": sorted(families),
        "retained_ids": [edge["candidate_id"] for edge in kept],
        "rejected": [{"candidate_id": edge["candidate_id"],
                      "reason": edge["rejection"],
                      "violations": edge.get("discovery", {}).get("violations")}
                     for edge in rejected],
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def audit(args: argparse.Namespace) -> None:
    graph = json.loads(args.graph.read_text())
    if not isinstance(graph, list):
        graph = graph.get("edges", graph.get("constraints", []))
    edges = [edge for edge in graph if edge.get("relation_family") in
             {"integer_domain", "domain_bound"}]
    rows = []
    per_edge = []
    for specification in args.sample:
        label, path = specification.split("=", 1)
        frame = pd.read_csv(path)
        any_bad = np.zeros(len(frame), dtype=bool)
        integer_bad = []
        bound_bad = []
        for edge in edges:
            bad = edge_violation(frame, edge)
            any_bad |= bad
            family = edge["relation_family"]
            (integer_bad if family == "integer_domain" else bound_bad).append(bad)
            per_edge.append({
                "method": label, "edge_id": edge["candidate_id"],
                "family": family, "column": edge["columns"][0],
                "violation_rate": float(bad.mean()), "violations": int(bad.sum()),
            })
        def macro(items: list[np.ndarray]) -> float:
            return float(np.mean([bad.mean() for bad in items])) if items else 0.0
        rows.append({
            "method": label, "rows": len(frame), "domain_edges": len(edges),
            "any_domain_violation_percent": 100 * float(any_bad.mean()),
            "integer_macro_violation_percent": 100 * macro(integer_bad),
            "bound_macro_violation_percent": 100 * macro(bound_bad),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(per_edge).to_csv(args.output / "per_edge.csv", index=False)
    columns = [
        "method", "rows", "domain_edges", "any_domain_violation_percent",
        "integer_macro_violation_percent", "bound_macro_violation_percent",
    ]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    table = "\n".join(lines)
    (args.output / "RESULTS.md").write_text(
        "# LCATS-style domain constraint audit\n\n" + table +
        "\n\nAll domain candidates were specified from schema semantics and retained "
        "only after zero-violation verification on the train split.\n")
    print(table)


def repair(args: argparse.Namespace) -> None:
    """Apply the exact LCATS-style endpoint domain operator.

    This is intentionally separate from the trajectory strength so an
    integer strength such as .25 cannot accidentally weaken the final-repair
    control.
    """
    graph = json.loads(args.graph.read_text())
    if not isinstance(graph, list):
        graph = graph.get("edges", graph.get("constraints", []))
    edges = [edge for edge in graph if edge.get("relation_family") in
             {"integer_domain", "domain_bound"}]
    frame = pd.read_csv(args.input)
    changed = np.zeros((len(frame), len(frame.columns)), dtype=bool)
    positions = {column: index for index, column in enumerate(frame.columns)}
    for edge in edges:
        if edge.get("program", {}).get("kind") == "integer":
            column = edge["program"]["column"]
            old = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            new = np.rint(old)
        else:
            atom = edge["requirements"][0]
            column = atom["left_column"]
            old = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            target = float(atom["right_value"])
            new = np.maximum(old, target) if atom["operator"] == ">=" else np.minimum(old, target)
        mask = np.isfinite(old) & (np.abs(new - old) > 1e-8)
        if mask.any():
            frame.loc[mask, column] = new[mask]
            changed[mask, positions[column]] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    diagnostics = {
        "rows": len(frame),
        "changed_rows": int(changed.any(axis=1).sum()),
        "changed_cells": int(changed.sum()),
        "changed_row_fraction": float(changed.any(axis=1).mean()),
        "changed_cell_fraction": float(changed.mean()),
    }
    args.output.with_suffix(".repair.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--dataset", default="news")
    builder.add_argument("--train", type=Path, required=True)
    builder.add_argument("--base-graph", type=Path)
    builder.add_argument("--families", default="integer,bounds")
    builder.add_argument("--output", type=Path, required=True)
    builder.set_defaults(function=build)
    auditor = sub.add_parser("audit")
    auditor.add_argument("--graph", type=Path, required=True)
    auditor.add_argument("--sample", action="append", required=True,
                         help="Label=/path/to/sample.csv; repeatable")
    auditor.add_argument("--output", type=Path, required=True)
    auditor.set_defaults(function=audit)
    fixer = sub.add_parser("repair")
    fixer.add_argument("--graph", type=Path, required=True)
    fixer.add_argument("--input", type=Path, required=True)
    fixer.add_argument("--output", type=Path, required=True)
    fixer.set_defaults(function=repair)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
