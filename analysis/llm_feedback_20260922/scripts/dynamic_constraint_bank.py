#!/usr/bin/env python3
"""Create a synthetic-only benchmark for conditional applicability and satisfaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CONSTRAINTS = {
    "husband_sex": {
        "description": "If relationship is Husband, sex must be Male.",
        "antecedent": {"relationship": "Husband"},
        "consequent": {"sex": "Male"},
        "columns": ["relationship", "sex"],
    },
    "wife_sex": {
        "description": "If relationship is Wife, sex must be Female.",
        "antecedent": {"relationship": "Wife"},
        "consequent": {"sex": "Female"},
        "columns": ["relationship", "sex"],
    },
    "spouse_marital": {
        "description": "If relationship is Husband or Wife, marital status must be Married-civ-spouse.",
        "antecedent": {"relationship_in": ["Husband", "Wife"]},
        "consequent": {"marital.status": "Married-civ-spouse"},
        "columns": ["relationship", "marital.status"],
    },
    "advanced_education_age": {
        "description": "If education is Doctorate or Prof-school, age must be at least 20.",
        "antecedent": {"education_in": ["Doctorate", "Prof-school"]},
        "consequent": {"age_gte": 20},
        "columns": ["education", "education.num", "age"],
    },
    "masters_age": {
        "description": "If education is Masters, age must be at least 18.",
        "antecedent": {"education": "Masters"},
        "consequent": {"age_gte": 18},
        "columns": ["education", "education.num", "age"],
    },
    "workclass_missingness": {
        "description": "If workclass is missing ('?'), occupation must also be missing ('?').",
        "antecedent": {"workclass": "?"},
        "consequent": {"occupation": "?"},
        "columns": ["workclass", "occupation"],
    },
    "occupation_missingness": {
        "description": "If occupation is missing ('?'), workclass must also be missing ('?').",
        "antecedent": {"occupation": "?"},
        "consequent": {"workclass": "?"},
        "columns": ["occupation", "workclass"],
    },
}


def setv(record: dict, column: str, value):
    record[column] = f" {value}" if isinstance(value, str) else value


def instantiate(record: dict, constraint_id: str, state: str) -> dict:
    r = dict(record)
    if constraint_id == "husband_sex":
        if state == "not_applicable":
            setv(r, "relationship", "Not-in-family"); setv(r, "sex", "Female")
        else:
            setv(r, "relationship", "Husband"); setv(r, "sex", "Female" if state == "violated" else "Male")
            setv(r, "marital.status", "Married-civ-spouse")
    elif constraint_id == "wife_sex":
        if state == "not_applicable":
            setv(r, "relationship", "Not-in-family"); setv(r, "sex", "Male")
        else:
            setv(r, "relationship", "Wife"); setv(r, "sex", "Male" if state == "violated" else "Female")
            setv(r, "marital.status", "Married-civ-spouse")
    elif constraint_id == "spouse_marital":
        if state == "not_applicable":
            setv(r, "relationship", "Not-in-family"); setv(r, "marital.status", "Never-married")
        else:
            setv(r, "relationship", "Husband"); setv(r, "sex", "Male")
            setv(r, "marital.status", "Never-married" if state == "violated" else "Married-civ-spouse")
    elif constraint_id == "advanced_education_age":
        if state == "not_applicable":
            setv(r, "education", "HS-grad"); r["education.num"] = 9.0; r["age"] = 14.0
        else:
            setv(r, "education", "Doctorate"); r["education.num"] = 16.0
            r["age"] = 14.0 if state == "violated" else 45.0
    elif constraint_id == "masters_age":
        if state == "not_applicable":
            setv(r, "education", "HS-grad"); r["education.num"] = 9.0; r["age"] = 16.0
        else:
            setv(r, "education", "Masters"); r["education.num"] = 14.0
            r["age"] = 16.0 if state == "violated" else 45.0
    elif constraint_id == "workclass_missingness":
        if state == "not_applicable":
            setv(r, "workclass", "Private"); setv(r, "occupation", "Exec-managerial")
        else:
            setv(r, "workclass", "?"); setv(r, "occupation", "Exec-managerial" if state == "violated" else "?")
    elif constraint_id == "occupation_missingness":
        if state == "not_applicable":
            setv(r, "occupation", "Exec-managerial"); setv(r, "workclass", "Private")
        else:
            setv(r, "occupation", "?"); setv(r, "workclass", "Private" if state == "violated" else "?")
    else:
        raise KeyError(constraint_id)
    return r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    clean = [case for case in source if case["control_type"] == "paired_clean"]
    times = sorted({float(case["t"]) for case in clean}, reverse=True)
    cases = []
    for t_value in times:
        pool = [case for case in clean if float(case["t"]) == t_value]
        cursor = 0
        for constraint_id, constraint in CONSTRAINTS.items():
            for state in ("violated", "satisfied", "not_applicable"):
                for repeat in range(args.replicates):
                    base = pool[cursor % len(pool)]
                    cursor += 1
                    record = instantiate(base["record"], constraint_id, state)
                    applicability = "not_applicable" if state == "not_applicable" else "applicable"
                    status = "violated" if state == "violated" else ("satisfied" if state == "satisfied" else "not_applicable")
                    case_id = f"dyn_{constraint_id}_{state}_t{t_value:.3f}_r{repeat}"
                    cases.append({
                        "case_id": case_id,
                        "dataset": "adult",
                        "t": t_value,
                        "trajectory_id": base["trajectory_id"],
                        "record": record,
                        "previous_record": base.get("previous_record"),
                        "constraint_id": constraint_id,
                        "constraint_description": constraint["description"],
                        "structured_predicate": {"antecedent": constraint["antecedent"], "consequent": constraint["consequent"]},
                        "involved_columns": constraint["columns"],
                        "gold_applicability": applicability,
                        "gold_status": status,
                        "state": state,
                        "privacy": "synthetic_preview_only",
                    })
    with (args.output / "dynamic_cases.jsonl").open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")
    summary = pd.DataFrame(cases).groupby(["t", "constraint_id", "state"]).size().rename("count").reset_index()
    summary.to_csv(args.output / "bank_summary.csv", index=False)
    metadata = {
        "status": "executed", "cases": len(cases), "constraints": list(CONSTRAINTS),
        "times": times, "replicates": args.replicates,
        "scope": "synthetic previews only; explicit applicable/satisfied/violated/not-applicable states",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
