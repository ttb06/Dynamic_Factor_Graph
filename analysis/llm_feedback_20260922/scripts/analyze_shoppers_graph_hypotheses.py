#!/usr/bin/env python3
"""Evaluate several graph-acquisition hypotheses from one budgeted Shoppers run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(path: Path):
    return json.loads(path.read_text())


def column_comparison_count(items: list[dict]) -> int:
    return sum(
        any(atom.get("right_kind") == "column"
            for atom in item.get("applicability", []) + item.get("requirements", [])
            + [a for group in item.get("exceptions", []) for a in group.get("all_of", [])])
        for item in items
    )


def stable(items: list[dict], threshold: float = .05) -> int:
    return sum(float(item.get("heldout_shift", np.inf)) <= threshold for item in items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    metadata = load(args.run / "run_metadata.json")
    protocol = load(args.run / "llm_protocol_graph.json")
    single = load(args.run / "strong_single_pass_graph.json")
    single_unjudged = load(args.run / "strong_single_pass_unjudged_graph.json")
    empirical = load(args.run / "empirical_graph.json")
    unified = load(args.run / "unified_graph.json")
    comparison = pd.read_csv(args.run / "graph_comparison.csv")

    protocol_keys = {x["canonical"] for x in protocol}
    single_keys = {x["canonical"] for x in single}
    empirical_keys = {x["canonical"] for x in empirical}
    round1 = [x for x in protocol if str(x.get("candidate_id", "")).startswith("r1_")]
    round2 = [x for x in protocol if str(x.get("candidate_id", "")).startswith("r2_")]
    seeds = [x for x in protocol if str(x.get("candidate_id", "")).startswith("seed_")]
    external = [x for x in protocol if x.get("evidence_basis") in {"external_domain", "hybrid"}]
    external_incremental = [x for x in external if x["canonical"] not in empirical_keys]

    def mean_shift(items):
        values = [float(x["heldout_shift"]) for x in items if np.isfinite(x.get("heldout_shift", np.nan))]
        return float(np.mean(values)) if values else np.nan

    rows = [
        {
            "hypothesis": "H1_multistage_vs_judged_single_pass",
            "primary_measure": "stable unique protocol edges versus judged single-pass",
            "value": len([x for x in protocol if x["canonical"] not in single_keys and x.get("heldout_shift", 1) <= .05]),
            "secondary": f"protocol={len(protocol)}, single={len(single)}, mean_shift={mean_shift(protocol):.4f}/{mean_shift(single):.4f}",
        },
        {
            "hypothesis": "H2_counterexample_refinement",
            "primary_measure": "retained stable round-2 edges",
            "value": stable(round2),
            "secondary": f"round1={len(round1)}, round2={len(round2)}, round2_mean_shift={mean_shift(round2):.4f}",
        },
        {
            "hypothesis": "H3_rich_typed_dsl",
            "primary_measure": "retained edges containing column comparisons",
            "value": column_comparison_count(protocol),
            "secondary": f"unified_column_comparisons={column_comparison_count(unified)}",
        },
        {
            "hypothesis": "H4_external_incremental_value",
            "primary_measure": "stable external/hybrid edges absent from empirical graph",
            "value": stable(external_incremental),
            "secondary": f"external={len(external)}, incremental={len(external_incremental)}",
        },
    ]
    hypotheses = pd.DataFrame(rows)
    hypotheses.to_csv(args.output / "hypothesis_summary.csv", index=False)

    edge_rows = []
    for graph, items in [("protocol", protocol), ("judged_single_pass", single),
                         ("unjudged_single_pass", single_unjudged), ("empirical", empirical),
                         ("unified", unified)]:
        for item in items:
            edge_rows.append({
                "graph": graph, "candidate_id": item.get("candidate_id"),
                "description": item.get("description"), "classification": item.get("classification"),
                "family": item.get("relation_family"), "evidence": item.get("evidence_basis"),
                "generalizability": item.get("generalizability"),
                "heldout_shift": item.get("heldout_shift"),
                "heldout_violation_rate": item.get("heldout", {}).get("violation_rate"),
                "column_comparison": column_comparison_count([item]) > 0,
            })
    pd.DataFrame(edge_rows).to_csv(args.output / "all_retained_edges.csv", index=False)

    lines = [
        "# Budgeted Shoppers knowledge-graph hypotheses", "",
        f"- Models: `{metadata['models']}`",
        f"- Token usage: **{metadata['usage'].get('total_tokens', 0):,} / {metadata.get('token_budget', 0):,}**",
        f"- API calls: **{metadata['api_calls']}**", "",
        "| Hypothesis | Primary measure | Value | Detail |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['hypothesis']} | {row['primary_measure']} | {row['value']} | {row['secondary']} |")
    lines += ["", "## Graph comparison", "",
              "| Graph | Proposed | Retained | Hard | Near-hard | Soft | Families | Mean heldout shift |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison.itertuples(index=False):
        lines.append(f"| {row.graph} | {row.proposed} | {row.retained} | {row.hard} | {row.near_hard} | "
                     f"{row.soft} | {row.family_diversity} | {row.mean_heldout_shift:.4f} |")
    lines += ["", "## Guardrails", "",
              "- A nonzero value is only preliminary support; there is no complete semantic gold graph.",
              "- Heldout stability is necessary but does not prove semantic truth.",
              "- H1 uses the same critic and judge for single-pass and multi-stage candidates.",
              "- H2 is an within-run provenance analysis; it avoids comparing two independently sampled API runs.",
              "- No source row was sent to the API."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
