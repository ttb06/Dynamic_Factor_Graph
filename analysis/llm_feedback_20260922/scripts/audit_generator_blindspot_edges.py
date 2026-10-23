#!/usr/bin/env python3
"""Rank frozen source-graph edges by generator-specific excess violation.

This is a diagnostic, not a new selection rule: it asks whether graph curation
discarded semantically reviewed relations that the generator actually fails.
Only aggregate calibration metadata and synthetic samples are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from typed_dynamic_graph_guidance import edge_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    graph = json.loads(args.graph.read_text())
    frame = pd.read_csv(args.samples)
    decisions = {x["candidate_id"]: x for x in json.loads(args.decisions.read_text())}
    rows = []
    for edge in graph:
        active, _, violation = edge_masks(frame, edge)
        support = int(active.sum())
        violations = int(violation.sum())
        generated_rate = violations / support if support else float("nan")
        calibration = edge.get("calibration", {})
        calibration_rate = float(calibration.get("violation_rate", float("nan")))
        expected = round(calibration_rate * support) if support else 0
        decision = decisions.get(edge["candidate_id"], {})
        semantic_authority = (
            edge.get("source") != "empirical_mining"
            and edge.get("classification") in {"hard", "near_hard"}
            and float(edge.get("judge_confidence", 0.0)) >= 0.65
        )
        rows.append({
            "candidate_id": edge["candidate_id"],
            "source": edge.get("source"),
            "classification": edge.get("classification"),
            "judge_confidence": edge.get("judge_confidence"),
            "curation_decision": decision.get("decision", "unknown"),
            "curation_reason": decision.get("reason", "unknown"),
            "synthetic_support": support,
            "synthetic_violations": violations,
            "synthetic_violation_rate": generated_rate,
            "calibration_violation_rate": calibration_rate,
            "generator_excess_rate": generated_rate - calibration_rate,
            "generator_excess_count": max(0, violations - expected),
            "semantic_actuation_authority": semantic_authority,
            "description": edge.get("description", ""),
        })
    result = pd.DataFrame(rows).sort_values(
        ["semantic_actuation_authority", "generator_excess_count", "generator_excess_rate"],
        ascending=[False, False, False],
    )
    result.to_csv(args.output / "generator_blindspot_edges.csv", index=False)
    actionable = result[
        result.semantic_actuation_authority
        & result.generator_excess_count.gt(0)
        & ~result.curation_decision.eq("actuate")
    ]
    summary = {
        "source_edges": len(result),
        "semantic_authority_edges": int(result.semantic_actuation_authority.sum()),
        "omitted_semantic_blindspots": int(len(actionable)),
        "omitted_excess_actions": int(actionable.generator_excess_count.sum()),
        "real_rows_read": False,
        "synthetic_rows": len(frame),
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Generator-blindspot audit", "", f"Summary: `{summary}`", "",
             "| Edge | Decision | Reason | Support | Violations | Generator rate | Calibration rate | Excess count |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in actionable.head(20).itertuples(index=False):
        lines.append(f"| {row.candidate_id} | {row.curation_decision} | {row.curation_reason} | "
                     f"{row.synthetic_support} | {row.synthetic_violations} | "
                     f"{row.synthetic_violation_rate:.4f} | {row.calibration_violation_rate:.4f} | "
                     f"{row.generator_excess_count} |")
    if actionable.empty:
        lines.append("| — | — | No omitted semantically authorized generator blindspot | — | — | — | — | — |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
