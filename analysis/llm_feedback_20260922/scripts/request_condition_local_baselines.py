#!/usr/bin/env python3
"""Oracle-DSL and deterministic template-parser baselines for request conditions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from request_condition_common import action_for_status, evaluate_dsl, make_predicate, make_repair
from request_condition_probe import decision_summary, normalized_dsl


def parse_template(text: str) -> dict:
    """Parse only the controlled request grammar used in the pilot bank.

    This intentionally strong baseline defines the scope where an LLM is not
    needed.  It should fail loudly rather than silently invent semantics.
    """

    age = re.search(r"education is Doctorate, Prof-school, or Masters, age must be at least (\d+)", text)
    if age:
        threshold = float(age.group(1))
        return {
            "antecedent_all": [make_predicate("education", "in", string_list=["Doctorate", "Prof-school", "Masters"])],
            "requirement_all": [make_predicate("age", "gte", number_value=threshold)],
            "exception_all": [],
            "repair": make_repair("age", number_value=threshold),
        }
    hours = re.search(r"workclass is Private, hours\.per\.week must be at least (\d+)", text)
    if hours:
        threshold = float(hours.group(1))
        return {
            "antecedent_all": [make_predicate("workclass", "eq", string_value="Private")],
            "requirement_all": [make_predicate("hours.per.week", "gte", number_value=threshold)],
            "exception_all": [],
            "repair": make_repair("hours.per.week", number_value=threshold),
        }
    if "relationship is Husband or Wife" in text and "Married-AF-spouse" in text:
        return {
            "antecedent_all": [make_predicate("relationship", "in", string_list=["Husband", "Wife"])],
            "requirement_all": [make_predicate("marital.status", "eq", string_value="Married-civ-spouse")],
            "exception_all": [make_predicate("marital.status", "eq", string_value="Married-AF-spouse")],
            "repair": make_repair("marital.status", string_value="Married-civ-spouse"),
        }
    if "occupation is '?'" in text and "workclass=Never-worked" in text:
        return {
            "antecedent_all": [make_predicate("occupation", "eq", string_value="?")],
            "requirement_all": [make_predicate("workclass", "eq", string_value="?")],
            "exception_all": [make_predicate("workclass", "eq", string_value="Never-worked")],
            "repair": make_repair("workclass", string_value="?"),
        }
    raise ValueError("request is outside the deterministic template grammar")


def evaluate(name: str, cases: list[dict], requests: list[dict], dsls: dict[str, dict], output: Path) -> dict:
    rows = []
    for case in cases:
        dsl = dsls.get(case["request_id"])
        status = evaluate_dsl(case["record"], dsl) if dsl is not None else "missing"
        rows.append(
            {
                "case_id": case["case_id"], "request_id": case["request_id"], "pair_id": case.get("pair_id"),
                "t": case["t"], "design": case["design"], "gold_status": case["gold_status"], "pred_status": status,
                "gold_action": case["gold_action"], "pred_action": action_for_status(status) if status != "missing" else "missing",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{name}_decisions.csv", index=False)
    summary = decision_summary(frame, mode=name, model="local_deterministic", usage={}, latency=0.0)
    summary["execution_equivalence"] = float((frame.gold_status == frame.pred_status).mean())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    requests = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    oracle = {request["request_id"]: request["oracle_dsl"] for request in requests}
    parsed, errors = {}, {}
    for request in requests:
        try:
            parsed[request["request_id"]] = parse_template(request["text"])
        except Exception as exc:
            errors[request["request_id"]] = f"{type(exc).__name__}: {exc}"
    summaries = {
        "oracle_dsl": evaluate("oracle_dsl", cases, requests, oracle, args.output),
        "template_parser": evaluate("template_parser", cases, requests, parsed, args.output),
    }
    summaries["template_parser"]["request_parse_rate"] = len(parsed) / len(requests)
    summaries["template_parser"]["dsl_structural_exact_match_rate"] = sum(
        normalized_dsl(parsed[request_id]) == normalized_dsl(oracle[request_id]) for request_id in parsed
    ) / len(requests)
    result = {
        "status": "executed",
        "baselines": summaries,
        "parse_errors": errors,
        "interpretation": "Template parser is an upper baseline for this controlled grammar, not a general NL parser.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
