#!/usr/bin/env python3
"""Build a leakage-controlled linguistic stress test for NL -> typed contracts.

All records are modifications of already synthetic TabDiff previews. The API
runner receives only the rendered request and synthetic record; oracle DSLs,
semantic family labels, gold statuses, and original IDs remain local.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from request_condition_common import evaluate_dsl, make_predicate, make_repair


def contract_catalog() -> list[dict]:
    advanced = ["Masters", "Prof-school", "Doctorate"]
    return [
        {
            "id": "advanced_age_21", "a": "education is Masters, Prof-school, or Doctorate",
            "r": "age is at least 21", "e": "occupation is Exec-managerial", "fix": "set age to 21",
            "dsl": {"antecedent_all": [make_predicate("education", "in", string_list=advanced)],
                    "requirement_all": [make_predicate("age", "gte", number_value=21)],
                    "exception_all": [make_predicate("occupation", "eq", string_value="Exec-managerial")],
                    "repair": make_repair("age", number_value=21)},
            "rows": {"satisfied": {"education":"Masters", "occupation":"Adm-clerical", "age":30},
                     "violated": {"education":"Doctorate", "occupation":"Sales", "age":18},
                     "not_applicable": {"education":"HS-grad", "occupation":"Sales", "age":18},
                     "excepted": {"education":"Prof-school", "occupation":"Exec-managerial", "age":18}},
        },
        {
            "id": "advanced_age_35", "a": "education belongs to {Masters, Prof-school, Doctorate}",
            "r": "age is greater than or equal to 35", "e": "occupation equals Exec-managerial", "fix": "replace age by 35",
            "dsl": {"antecedent_all": [make_predicate("education", "in", string_list=advanced)],
                    "requirement_all": [make_predicate("age", "gte", number_value=35)],
                    "exception_all": [make_predicate("occupation", "eq", string_value="Exec-managerial")],
                    "repair": make_repair("age", number_value=35)},
            "rows": {"satisfied": {"education":"Doctorate", "occupation":"Sales", "age":40},
                     "violated": {"education":"Masters", "occupation":"Adm-clerical", "age":30},
                     "not_applicable": {"education":"Bachelors", "occupation":"Sales", "age":30},
                     "excepted": {"education":"Masters", "occupation":"Exec-managerial", "age":30}},
        },
        {
            "id": "private_hours_30", "a": "workclass is Private", "r": "hours.per.week is at least 30",
            "e": "occupation is Armed-Forces", "fix": "set hours.per.week to 30",
            "dsl": {"antecedent_all": [make_predicate("workclass", "eq", string_value="Private")],
                    "requirement_all": [make_predicate("hours.per.week", "gte", number_value=30)],
                    "exception_all": [make_predicate("occupation", "eq", string_value="Armed-Forces")],
                    "repair": make_repair("hours.per.week", number_value=30)},
            "rows": {"satisfied": {"workclass":"Private", "occupation":"Sales", "hours.per.week":40},
                     "violated": {"workclass":"Private", "occupation":"Sales", "hours.per.week":20},
                     "not_applicable": {"workclass":"State-gov", "occupation":"Sales", "hours.per.week":20},
                     "excepted": {"workclass":"Private", "occupation":"Armed-Forces", "hours.per.week":20}},
        },
        {
            "id": "private_hours_45", "a": "workclass equals Private", "r": "hours.per.week is no lower than 45",
            "e": "occupation equals Armed-Forces", "fix": "change hours.per.week to 45",
            "dsl": {"antecedent_all": [make_predicate("workclass", "eq", string_value="Private")],
                    "requirement_all": [make_predicate("hours.per.week", "gte", number_value=45)],
                    "exception_all": [make_predicate("occupation", "eq", string_value="Armed-Forces")],
                    "repair": make_repair("hours.per.week", number_value=45)},
            "rows": {"satisfied": {"workclass":"Private", "occupation":"Sales", "hours.per.week":50},
                     "violated": {"workclass":"Private", "occupation":"Sales", "hours.per.week":35},
                     "not_applicable": {"workclass":"Local-gov", "occupation":"Sales", "hours.per.week":35},
                     "excepted": {"workclass":"Private", "occupation":"Armed-Forces", "hours.per.week":35}},
        },
        {
            "id": "advanced_private_hours_cap", "a": "education is Masters or Doctorate and workclass is Private",
            "r": "hours.per.week is at most 50", "e": "occupation is Exec-managerial", "fix": "set hours.per.week to 50",
            "dsl": {"antecedent_all": [make_predicate("education", "in", string_list=["Masters","Doctorate"]), make_predicate("workclass", "eq", string_value="Private")],
                    "requirement_all": [make_predicate("hours.per.week", "lte", number_value=50)],
                    "exception_all": [make_predicate("occupation", "eq", string_value="Exec-managerial")],
                    "repair": make_repair("hours.per.week", number_value=50)},
            "rows": {"satisfied": {"education":"Masters", "workclass":"Private", "occupation":"Sales", "hours.per.week":45},
                     "violated": {"education":"Doctorate", "workclass":"Private", "occupation":"Sales", "hours.per.week":60},
                     "not_applicable": {"education":"HS-grad", "workclass":"Private", "occupation":"Sales", "hours.per.week":60},
                     "excepted": {"education":"Masters", "workclass":"Private", "occupation":"Exec-managerial", "hours.per.week":60}},
        },
        {
            "id": "sales_tech_gain", "a": "occupation is Sales or Tech-support", "r": "capital.gain is at least 500",
            "e": "workclass is Self-emp-inc", "fix": "set capital.gain to 500",
            "dsl": {"antecedent_all": [make_predicate("occupation", "in", string_list=["Sales","Tech-support"])],
                    "requirement_all": [make_predicate("capital.gain", "gte", number_value=500)],
                    "exception_all": [make_predicate("workclass", "eq", string_value="Self-emp-inc")],
                    "repair": make_repair("capital.gain", number_value=500)},
            "rows": {"satisfied": {"occupation":"Sales", "workclass":"Private", "capital.gain":1000},
                     "violated": {"occupation":"Tech-support", "workclass":"Private", "capital.gain":0},
                     "not_applicable": {"occupation":"Craft-repair", "workclass":"Private", "capital.gain":0},
                     "excepted": {"occupation":"Sales", "workclass":"Self-emp-inc", "capital.gain":0}},
        },
        {
            "id": "non_us_hours_cap", "a": "native.country is neither United-States nor Canada",
            "r": "hours.per.week is at most 45", "e": "workclass is Federal-gov", "fix": "set hours.per.week to 45",
            "dsl": {"antecedent_all": [make_predicate("native.country", "not_in", string_list=["United-States","Canada"])],
                    "requirement_all": [make_predicate("hours.per.week", "lte", number_value=45)],
                    "exception_all": [make_predicate("workclass", "eq", string_value="Federal-gov")],
                    "repair": make_repair("hours.per.week", number_value=45)},
            "rows": {"satisfied": {"native.country":"Mexico", "workclass":"Private", "hours.per.week":40},
                     "violated": {"native.country":"Germany", "workclass":"Private", "hours.per.week":60},
                     "not_applicable": {"native.country":"United-States", "workclass":"Private", "hours.per.week":60},
                     "excepted": {"native.country":"Mexico", "workclass":"Federal-gov", "hours.per.week":60}},
        },
        {
            "id": "spouse_marital", "a": "relationship is Husband or Wife", "r": "marital.status is Married-civ-spouse",
            "e": "marital.status is Married-AF-spouse", "fix": "set marital.status to Married-civ-spouse",
            "dsl": {"antecedent_all": [make_predicate("relationship", "in", string_list=["Husband","Wife"])],
                    "requirement_all": [make_predicate("marital.status", "eq", string_value="Married-civ-spouse")],
                    "exception_all": [make_predicate("marital.status", "eq", string_value="Married-AF-spouse")],
                    "repair": make_repair("marital.status", string_value="Married-civ-spouse")},
            "rows": {"satisfied": {"relationship":"Wife", "marital.status":"Married-civ-spouse"},
                     "violated": {"relationship":"Husband", "marital.status":"Never-married"},
                     "not_applicable": {"relationship":"Own-child", "marital.status":"Never-married"},
                     "excepted": {"relationship":"Husband", "marital.status":"Married-AF-spouse"}},
        },
        {
            "id": "missing_workclass", "a": "occupation is ?", "r": "workclass is ?",
            "e": "workclass is Never-worked", "fix": "set workclass to ?",
            "dsl": {"antecedent_all": [make_predicate("occupation", "eq", string_value="?")],
                    "requirement_all": [make_predicate("workclass", "eq", string_value="?")],
                    "exception_all": [make_predicate("workclass", "eq", string_value="Never-worked")],
                    "repair": make_repair("workclass", string_value="?")},
            "rows": {"satisfied": {"occupation":"?", "workclass":"?"},
                     "violated": {"occupation":"?", "workclass":"Private"},
                     "not_applicable": {"occupation":"Sales", "workclass":"Private"},
                     "excepted": {"occupation":"?", "workclass":"Never-worked"}},
        },
        {
            "id": "hs_grad_code", "a": "education is HS-grad", "r": "education.num equals 9",
            "e": "occupation is ?", "fix": "set education.num to 9",
            "dsl": {"antecedent_all": [make_predicate("education", "eq", string_value="HS-grad")],
                    "requirement_all": [make_predicate("education.num", "eq", number_value=9)],
                    "exception_all": [make_predicate("occupation", "eq", string_value="?")],
                    "repair": make_repair("education.num", number_value=9)},
            "rows": {"satisfied": {"education":"HS-grad", "education.num":9, "occupation":"Sales"},
                     "violated": {"education":"HS-grad", "education.num":13, "occupation":"Sales"},
                     "not_applicable": {"education":"Bachelors", "education.num":13, "occupation":"Sales"},
                     "excepted": {"education":"HS-grad", "education.num":13, "occupation":"?"}},
        },
    ]


def surfaces(contract: dict) -> list[str]:
    values = {"a": contract["a"], "r": contract["r"], "e": contract["e"], "fix": contract["fix"]}
    templates = [
        "If {a}, require that {r}. Exception: {e}. For a true violation, {fix}.",
        "Apply this policy only to records where {a}: they must have {r}, unless {e}; if the rule fails, {fix}.",
        "First exempt a record whenever {e}. Among all remaining records for which {a}, enforce that {r}; {fix} when needed.",
        "Policy formula: ({a}) AND NOT ({e}) IMPLIES ({r}). The remediation is to {fix}.",
        "A row enters the policy scope when {a}. It is compliant exactly when {r}. However, label it excepted rather than invalid if {e}. Remediate only an actual breach: {fix}.",
    ]
    return [template.format(**values) for template in templates]


def apply_overrides(record: dict, overrides: dict) -> dict:
    result = dict(record)
    for column, value in overrides.items():
        result[column] = f" {value}" if isinstance(value, str) else value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = [json.loads(line) for line in args.source_cases.read_text().splitlines() if line.strip()]
    if not source or any(case.get("privacy") != "synthetic_preview_only" for case in source):
        raise RuntimeError("source must contain marked synthetic previews only")

    requests, cases = [], []
    cursor = 0
    for contract in contract_catalog():
        for surface_id, text in enumerate(surfaces(contract)):
            request_id = f"{contract['id']}_p{surface_id}"
            requests.append({
                "request_id": request_id,
                "family": contract["id"],
                "surface_id": surface_id,
                "text": text,
                "oracle_dsl": contract["dsl"],
            })
            for status, overrides in contract["rows"].items():
                for replicate in range(args.replicates):
                    source_case = source[cursor % len(source)]
                    cursor += 1
                    row = apply_overrides(source_case["record"], overrides)
                    observed = evaluate_dsl(row, contract["dsl"])
                    if observed != status:
                        raise AssertionError((contract["id"], status, observed, row))
                    cases.append({
                        "case_id": f"lingcase_{len(cases):05d}",
                        "row_id": f"{contract['id']}_{status}_{replicate}",
                        "pair_id": None,
                        "trajectory_id": source_case.get("trajectory_id", "synthetic_source"),
                        "t": float(source_case.get("t", 0.0)),
                        "dataset": "adult",
                        "record": row,
                        "request_id": request_id,
                        "request_family": contract["id"],
                        "request_text": text,
                        "design": f"surface_{surface_id}",
                        "gold_status": status,
                        "gold_action": "repair_candidate" if status == "violated" else "no_action",
                        "privacy": "synthetic_preview_only",
                    })

    for filename, rows in (("requests.jsonl", requests), ("cases.jsonl", cases)):
        with (args.output / filename).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    vocabulary = {
        "education": ["HS-grad","Bachelors","Masters","Prof-school","Doctorate"],
        "occupation": ["?","Adm-clerical","Sales","Tech-support","Craft-repair","Exec-managerial","Armed-Forces"],
        "workclass": ["?","Private","State-gov","Local-gov","Federal-gov","Self-emp-inc","Never-worked"],
        "relationship": ["Husband","Wife","Own-child"],
        "marital.status": ["Married-civ-spouse","Married-AF-spouse","Never-married"],
        "native.country": ["United-States","Canada","Mexico","Germany"],
    }
    (args.output / "categorical_vocabulary.json").write_text(json.dumps(vocabulary, indent=2))
    metadata = {
        "status": "executed", "contracts": len(contract_catalog()), "surface_forms_per_contract": 5,
        "requests": len(requests), "cases": len(cases), "replicates_per_status_surface": args.replicates,
        "status_counts": dict(Counter(case["gold_status"] for case in cases)),
        "source_policy": "Only marked synthetic TabDiff previews; no real rows.",
        "leakage_policy": "API runner replaces case/request IDs; oracle DSL, family, surface and gold remain local.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
