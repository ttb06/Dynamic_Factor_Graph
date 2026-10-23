#!/usr/bin/env python3
"""Counterexample-test compiled contracts beyond the finite evaluation bank."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from request_condition_common import evaluate_dsl, predicate_value


def set_true(row: dict, predicate: dict, vocabulary: dict[str, list[str]]) -> list[dict]:
    column, operator, value = predicate["column"], predicate["operator"], predicate_value(predicate)
    values = value if operator == "in" else [value]
    if operator == "not_in":
        values = [candidate for candidate in vocabulary.get(column, []) if candidate not in value][:2]
    if operator == "neq":
        if isinstance(value, str):
            values = [candidate for candidate in vocabulary.get(column, []) if candidate != value][:2]
        else:
            values = [float(value) + 1]
    if operator in {"gte", "lte"}:
        values = [float(value)]
    if operator == "gt":
        values = [float(value) + 1]
    if operator == "lt":
        values = [float(value) - 1]
    result = []
    for candidate in values:
        changed = dict(row); changed[column] = candidate; result.append(changed)
    return result


def set_false(row: dict, predicate: dict, vocabulary: dict[str, list[str]]) -> dict:
    column, operator, value = predicate["column"], predicate["operator"], predicate_value(predicate)
    changed = dict(row)
    if operator == "eq":
        candidates = [candidate for candidate in vocabulary.get(column, []) if candidate != value]
        changed[column] = candidates[0] if candidates else (float(value) + 1 if not isinstance(value, str) else "__other__")
    elif operator == "neq":
        changed[column] = value
    elif operator == "in":
        candidates = [candidate for candidate in vocabulary.get(column, []) if candidate not in value]
        changed[column] = candidates[0] if candidates else "__other__"
    elif operator == "not_in":
        changed[column] = value[0]
    elif operator in {"gte", "gt"}:
        changed[column] = float(value) - 1
    elif operator in {"lte", "lt"}:
        changed[column] = float(value) + 1
    return changed


def true_antecedent_rows(seed: dict, predicates: list[dict], vocabulary: dict[str, list[str]]) -> list[dict]:
    rows = [dict(seed)]
    for predicate in predicates:
        rows = list(itertools.chain.from_iterable(set_true(row, predicate, vocabulary) for row in rows))
    return rows


def generated_cases(seed: dict, dsl: dict, vocabulary: dict[str, list[str]]) -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []
    antecedent_rows = true_antecedent_rows(seed, dsl["antecedent_all"], vocabulary)
    for ordinal, base in enumerate(antecedent_rows):
        satisfied = [dict(base)]
        for predicate in dsl["requirement_all"]:
            satisfied = list(itertools.chain.from_iterable(set_true(row, predicate, vocabulary) for row in satisfied))
        cases.extend((f"antecedent_variant_{ordinal}_satisfied", row) for row in satisfied)
        violated = dict(base)
        violated = set_false(violated, dsl["requirement_all"][0], vocabulary)
        cases.append((f"antecedent_variant_{ordinal}_violated", violated))
    not_applicable = dict(antecedent_rows[0])
    not_applicable = set_false(not_applicable, dsl["antecedent_all"][0], vocabulary)
    cases.append(("not_applicable", not_applicable))
    if dsl["exception_all"]:
        exception_rows = [dict(antecedent_rows[0])]
        for predicate in dsl["exception_all"]:
            exception_rows = list(itertools.chain.from_iterable(set_true(row, predicate, vocabulary) for row in exception_rows))
        cases.extend(("excepted", row) for row in exception_rows)
    return cases


def invalid_literals(dsl: dict, vocabulary: dict[str, list[str]]) -> list[dict]:
    invalid = []
    for section in ("antecedent_all", "requirement_all", "exception_all"):
        for predicate in dsl[section]:
            column = predicate["column"]
            if column not in vocabulary or predicate["value_type"] not in {"string", "string_list"}:
                continue
            values = predicate_value(predicate)
            values = values if isinstance(values, list) else [values]
            for value in values:
                if value not in vocabulary[column]:
                    invalid.append({"section": section, "column": column, "literal": value})
    return invalid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    requests = {item["request_id"]: item for item in (json.loads(line) for line in args.requests.read_text().splitlines() if line.strip())}
    vocabulary = json.loads(args.vocabulary.read_text())
    seed_by_request = {}
    for case in cases:
        seed_by_request.setdefault(case["request_id"], case["record"])

    summary_rows, counterexamples = [], []
    for compiled_path in args.compiled:
        frame = pd.read_csv(compiled_path)
        run = compiled_path.parent.name
        for _, compiled_row in frame.iterrows():
            request_id = compiled_row["request_id"]
            predicted = json.loads(compiled_row["predicted_dsl"])
            oracle = requests[request_id]["oracle_dsl"]
            audit_cases = generated_cases(seed_by_request[request_id], oracle, vocabulary)
            correct = 0
            for design, row in audit_cases:
                gold = evaluate_dsl(row, oracle)
                try:
                    prediction = evaluate_dsl(row, predicted)
                except Exception as exc:
                    prediction = f"error:{type(exc).__name__}"
                correct += prediction == gold
                if prediction != gold:
                    counterexamples.append(
                        {
                            "run": run, "request_id": request_id, "design": design,
                            "gold": gold, "prediction": prediction, "synthetic_record": json.dumps(row),
                        }
                    )
            invalid = invalid_literals(predicted, vocabulary)
            summary_rows.append(
                {
                    "run": run, "request_id": request_id, "counterexample_cases": len(audit_cases),
                    "semantic_equivalence_rate": correct / len(audit_cases),
                    "invalid_literal_count": len(invalid), "invalid_literals": json.dumps(invalid),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "per_request.csv", index=False)
    pd.DataFrame(counterexamples).to_csv(args.output / "counterexamples.csv", index=False)
    by_run = summary.groupby("run").agg(
        requests=("request_id", "size"), exact_semantic_requests=("semantic_equivalence_rate", lambda values: int((values == 1).sum())),
        mean_counterexample_accuracy=("semantic_equivalence_rate", "mean"), invalid_literals=("invalid_literal_count", "sum"),
    ).reset_index()
    by_run.to_csv(args.output / "by_run.csv", index=False)
    result = {
        "status": "executed", "runs": len(by_run), "counterexamples": len(counterexamples),
        "privacy": "All audited records are synthetic mutations; vocabulary is schema metadata only.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(result, indent=2))
    print(by_run.to_string(index=False)); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
