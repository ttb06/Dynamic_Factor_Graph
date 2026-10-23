#!/usr/bin/env python3
"""Extract a composable repayment-status chain from the verified Default graph."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


IDS = ["emp_000", "emp_001", "emp_002", "emp_005", "emp_012"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source.read_text())
    by_id = {edge["candidate_id"]: edge for edge in source}
    graph = []
    for candidate_id in IDS:
        edge = dict(by_id[candidate_id]); edge["usage_role"] = "actuate"
        graph.append(edge)
    expected = [
        ("PAY_0", "PAY_2"), ("PAY_2", "PAY_3"), ("PAY_3", "PAY_4"),
        ("PAY_4", "PAY_5"), ("PAY_5", "PAY_6"),
    ]
    observed = [(edge["applicability"][0]["left_column"], edge["requirements"][0]["left_column"])
                for edge in graph]
    if observed != expected:
        raise ValueError(f"source graph no longer matches expected chain: {observed}")
    (args.output / "chain.json").write_text(json.dumps(graph, indent=2))
    with (args.output / "schedule.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "candidate_id", "cutoff", "readable", "violation_evaluable"
        ])
        writer.writeheader()
        for edge in graph:
            writer.writerow({"candidate_id": edge["candidate_id"], "cutoff": .5,
                             "readable": True, "violation_evaluable": True})
    (args.output / "run_summary.json").write_text(json.dumps({
        "source": str(args.source), "candidate_ids": IDS,
        "status": "data-discovered and held-out-audited probabilistic chain",
        "hard_constraint": False, "edges": len(graph), "real_rows_emitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
