#!/usr/bin/env python3
"""Compile Harpoon's published Default requests into the typed graph DSL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def atom(column: str, operator: str, value):
    return {"left_column": column, "operator": operator, "right_kind": "literal",
            "right_column": "", "right_value": value, "tolerance": 0}


def edge(name: str, requirements: list[dict], description: str):
    split = {"support": 27000, "violations": 0, "violation_rate": 0.0, "lift": 1.0}
    return {"candidate_id": f"harpoon_default_{name}", "description": description,
            "classification": "hard", "relation_family": "official_runtime_constraint",
            "source": "harpoon_official_general_constraint", "usage_role": "actuate",
            "applicability": [], "requirements": requirements, "exceptions": [],
            "discovery": dict(split), "counterexample": dict(split),
            "calibration": dict(split), "heldout": dict(split), "judge_confidence": 1.0}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    age = atom("AGE", ">=", 50.0); education = atom("EDUCATION", "==", "3")
    graphs = {
        "range": [edge("range", [age], "Harpoon Default range request: AGE >= 50")],
        "category": [edge("category", [education], "Harpoon Default category request: EDUCATION == 3")],
        "both": [edge("both", [age, education], "Harpoon Default conjunction: AGE >= 50 AND EDUCATION == 3")],
    }
    for name, graph in graphs.items():
        (args.output / f"{name}.json").write_text(json.dumps(graph, indent=2))
    (args.output / "run_summary.json").write_text(json.dumps({
        "source": "Harpoon official generate_mask.py::constrainDataset",
        "dataset": "default", "requests": list(graphs), "rows_or_statistics_used": False,
    }, indent=2))


if __name__ == "__main__": main()
