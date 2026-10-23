#!/usr/bin/env python3
"""Build a controlled two-edge graph whose second edge emerges after the first action."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def atom(column: str, operator: str, value):
    return {"left_column": column, "operator": operator, "right_kind": "literal",
            "right_column": "", "right_value": value, "tolerance": 0}


def edge(candidate_id: str, description: str, applicability: list[dict], requirements: list[dict]):
    split = {"support": 27000, "violations": 0, "violation_rate": 0.0, "lift": 1.0}
    return {
        "candidate_id": candidate_id,
        "description": description,
        "classification": "hard",
        "relation_family": "controlled_dynamic_chain",
        "source": "controlled user-request mechanism probe",
        "usage_role": "actuate",
        "applicability": applicability,
        "requirements": requirements,
        "exceptions": [],
        "discovery": dict(split), "counterexample": dict(split),
        "calibration": dict(split), "heldout": dict(split),
        "judge_confidence": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    marriage = atom("MARRIAGE", "==", "1")
    age = atom("AGE", ">=", 50.0)
    graph = [
        edge("chain_age", "If MARRIAGE == 1, require AGE >= 50", [marriage], [age]),
        edge("chain_education", "If MARRIAGE == 1 and AGE >= 50, require EDUCATION == 3",
             [marriage, age], [atom("EDUCATION", "==", "3")]),
    ]
    (args.output / "chain.json").write_text(json.dumps(graph, indent=2))
    with (args.output / "schedule.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "candidate_id", "cutoff", "readable", "violation_evaluable"
        ])
        writer.writeheader()
        for item in graph:
            writer.writerow({"candidate_id": item["candidate_id"], "cutoff": .5,
                             "readable": True, "violation_evaluable": True})
    (args.output / "run_summary.json").write_text(json.dumps({
        "status": "controlled mechanism benchmark",
        "not_a_semantic_invariant": True,
        "request": "MARRIAGE == 1 -> AGE >= 50; then AGE >= 50 -> EDUCATION == 3 within scope",
        "mechanism": "the chain_education edge becomes applicable after chain_age actuation",
        "real_rows_emitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
