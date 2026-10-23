#!/usr/bin/env python3
"""Select executable hard/near-hard edges for row-wise injection experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORTED = {"==", "!=", ">", ">=", "<", "<="}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-judge-confidence", type=float, default=.65)
    parser.add_argument("--min-calibration-support", type=int, default=100)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.graph.read_text())
    selected, rejected = [], []
    for edge in source:
        reason = None
        if edge.get("classification") not in {"hard", "near_hard"}: reason = "soft_edge"
        elif float(edge.get("judge_confidence", 0)) < args.min_judge_confidence: reason = "low_judge_confidence"
        elif int(edge.get("calibration", {}).get("support", 0)) < args.min_calibration_support: reason = "low_support"
        elif not edge.get("requirements"): reason = "no_requirement"
        elif any(atom.get("operator") not in SUPPORTED for atom in edge.get("requirements", [])): reason = "unsupported_operator"
        if reason:
            rejected.append({"candidate_id": edge.get("candidate_id"), "description": edge.get("description"), "reason": reason})
        else:
            selected.append(edge)
    selected.sort(key=lambda x: ({"hard": 0, "near_hard": 1}[x["classification"]], -float(x.get("judge_confidence", 0))))
    (args.output / "active_graph.json").write_text(json.dumps(selected, indent=2))
    (args.output / "rejections.json").write_text(json.dumps(rejected, indent=2))
    complex_count = sum(
        len(x.get("applicability", [])) > 1 or len(x.get("requirements", [])) > 1 or bool(x.get("exceptions"))
        for x in selected
    )
    summary = {"source_edges": len(source), "selected_edges": len(selected), "complex_edges": complex_count,
               "non_seed_edges": sum(x.get("proposal_role") != "predefined_seed" for x in selected)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Row-wise mechanism graph", "", f"Summary: `{summary}`", "",
             "| ID | Class | Requirements | Exceptions | Role | Description |",
             "| --- | --- | ---: | ---: | --- | --- |"]
    for edge in selected:
        lines.append(f"| {edge['candidate_id']} | {edge['classification']} | {len(edge.get('requirements', []))} | "
                     f"{len(edge.get('exceptions', []))} | {edge.get('proposal_role')} | {edge['description']} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
