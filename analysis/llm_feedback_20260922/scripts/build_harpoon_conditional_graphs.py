#!/usr/bin/env python3
"""Create conditional extensions of Harpoon's public Default requests.

These are controlled user requests, not discovered real-world invariants.  They
hold the Harpoon consequents fixed while adding a row-dependent applicability
predicate, so dynamic versus frozen/shuffled topology can be tested directly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def atom(column: str, operator: str, value):
    return {"left_column": column, "operator": operator, "right_kind": "literal",
            "right_column": "", "right_value": value, "tolerance": 0}


def edge(name: str, requirements: list[dict]):
    split = {"support": 27000, "violations": 0, "violation_rate": 0.0, "lift": 1.0}
    return {
        "candidate_id": f"harpoon_conditional_{name}",
        "description": f"Controlled request: if MARRIAGE == 1, enforce Harpoon {name}",
        "classification": "hard",
        "relation_family": "conditional_public_request_extension",
        "source": "controlled extension of Harpoon Default request",
        "usage_role": "actuate",
        "applicability": [atom("MARRIAGE", "==", "1")],
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
    age = atom("AGE", ">=", 50.0)
    education = atom("EDUCATION", "==", "3")
    graphs = {
        "range": [edge("range", [age])],
        "category": [edge("category", [education])],
        "both": [edge("both", [age, education])],
    }
    for request, graph in graphs.items():
        (args.output / f"{request}.json").write_text(json.dumps(graph, indent=2))
    with (args.output / "schedule.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "cutoff", "readable", "violation_evaluable"])
        writer.writeheader()
        for request in graphs:
            writer.writerow({"candidate_id": f"harpoon_conditional_{request}", "cutoff": 0.5,
                             "readable": True, "violation_evaluable": True})
            writer.writerow({"candidate_id": f"harpoon_default_{request}", "cutoff": 0.5,
                             "readable": True, "violation_evaluable": True})
    (args.output / "run_summary.json").write_text(json.dumps({
        "status": "controlled benchmark extension",
        "not_a_semantic_invariant": True,
        "applicability": "MARRIAGE == 1",
        "consequents": ["AGE >= 50", "EDUCATION == 3", "conjunction"],
        "purpose": "isolate trajectory-dependent row applicability",
        "real_rows_emitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
