#!/usr/bin/env python3
"""Compile two public CuTS Adult implication programs into typed edges.

The CuTS predicates contain disjunctions.  We split each disjunctive antecedent
into an equivalent edge while retaining conjunctive requirements inside an
edge.  Only the repository training split is used for empirical verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def atom(column: str, operator: str, value: str) -> dict:
    return {
        "left_column": column,
        "operator": operator,
        "right_kind": "literal",
        "right_column": "",
        "right_value": value,
        "tolerance": 0.0,
    }


SPECS = [
    {
        "public_program": "implication2",
        "description": "IF marital.status == Widowed THEN sex == Female",
        "applicability": [atom("marital.status", "==", "Widowed")],
        "requirements": [atom("sex", "==", "Female")],
    },
    {
        "public_program": "implication2",
        "description": "IF relationship == Wife THEN sex == Female",
        "applicability": [atom("relationship", "==", "Wife")],
        "requirements": [atom("sex", "==", "Female")],
    },
    {
        "public_program": "implication4",
        "description": "IF marital.status == Divorced THEN relationship not in {Husband, Wife}",
        "applicability": [atom("marital.status", "==", "Divorced")],
        # This order also makes the sequential typed actuator end in a value
        # satisfying both exclusions for either forbidden starting category.
        "requirements": [atom("relationship", "!=", "Wife"), atom("relationship", "!=", "Husband")],
    },
    {
        "public_program": "implication4",
        "description": "IF marital.status == Never-married THEN relationship not in {Husband, Wife}",
        "applicability": [atom("marital.status", "==", "Never-married")],
        "requirements": [atom("relationship", "!=", "Wife"), atom("relationship", "!=", "Husband")],
    },
]


def mask(frame: pd.DataFrame, predicate: dict) -> np.ndarray:
    values = frame[predicate["left_column"]].astype("string").str.strip()
    result = values.eq(str(predicate["right_value"])).fillna(False).to_numpy()
    return ~result if predicate["operator"] == "!=" else result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(args.repo / "data/adult/train.csv")
    graph = []
    for index, spec in enumerate(SPECS):
        active = np.ones(len(train), dtype=bool)
        satisfied = np.ones(len(train), dtype=bool)
        for predicate in spec["applicability"]:
            active &= mask(train, predicate)
        for predicate in spec["requirements"]:
            satisfied &= mask(train, predicate)
        violations = int((active & ~satisfied).sum())
        support = int(active.sum())
        rate = violations / support if support else 0.0
        graph.append({
            "candidate_id": f"adult_cuts_{index:03d}",
            "description": spec["description"],
            "applicability": spec["applicability"],
            "requirements": spec["requirements"],
            "exceptions": [],
            "relation_family": "external_logical_implication",
            "hardness_prior": "hard",
            "classification": "hard",
            "source": "cuts_public_adult_program",
            "evidence_basis": "external_specification_verified_on_train",
            "provenance": ["CuTS public Adult program", "train_only_verification"],
            "usage_role": "actuate",
            "external_authority": True,
            "confidence": 1.0 - rate,
            "calibration": {
                "support": support,
                "violations": violations,
                "violation_rate": rate,
            },
            "public_program": spec["public_program"],
        })
    (args.output / "graph.json").write_text(json.dumps(graph, indent=2))
    metadata = {
        "dataset": "adult",
        "source": "CuTS public Adult implication programs 2 and 4",
        "programs": sorted({item["public_program"] for item in SPECS}),
        "edges": len(graph),
        "train_rows": len(train),
        "heldout_used": False,
        "compilation": "disjunctive antecedents split into equivalent typed edges",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    for edge in graph:
        print(edge["candidate_id"], edge["calibration"])


if __name__ == "__main__":
    main()
