#!/usr/bin/env python3
"""Shared typed-DSL evaluator for the request-conditioned constraint probe.

The evaluator is deliberately small and deterministic.  It is not an LLM
judge: an LLM may compile natural language into this representation, but all
gold labels and execution happen locally.
"""

from __future__ import annotations

from typing import Any


OPERATORS = {"eq", "neq", "in", "not_in", "gte", "lte", "gt", "lt"}
STATUSES = {"violated", "satisfied", "not_applicable", "excepted"}


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def predicate_value(predicate: dict) -> Any:
    value_type = predicate["value_type"]
    if value_type == "string":
        return normalize_scalar(predicate.get("string_value"))
    if value_type == "number":
        return float(predicate["number_value"])
    if value_type == "string_list":
        return [normalize_scalar(item) for item in predicate.get("string_list", [])]
    raise ValueError(f"unknown value_type={value_type!r}")


def evaluate_predicate(record: dict, predicate: dict) -> bool:
    operator = predicate["operator"]
    if operator not in OPERATORS:
        raise ValueError(f"unknown operator={operator!r}")
    if predicate["column"] not in record:
        raise KeyError(predicate["column"])
    observed = normalize_scalar(record[predicate["column"]])
    expected = predicate_value(predicate)
    if operator == "eq":
        return observed == expected
    if operator == "neq":
        return observed != expected
    if operator == "in":
        return observed in expected
    if operator == "not_in":
        return observed not in expected
    try:
        observed_number = float(observed)
        expected_number = float(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"non-numeric comparison for {predicate['column']}: {observed!r} {operator} {expected!r}"
        ) from exc
    return {
        "gte": observed_number >= expected_number,
        "lte": observed_number <= expected_number,
        "gt": observed_number > expected_number,
        "lt": observed_number < expected_number,
    }[operator]


def evaluate_all(record: dict, predicates: list[dict]) -> bool:
    return all(evaluate_predicate(record, predicate) for predicate in predicates)


def validate_dsl(dsl: dict) -> None:
    for key in ("antecedent_all", "requirement_all", "exception_all", "repair"):
        if key not in dsl:
            raise ValueError(f"DSL missing {key}")
    if not dsl["antecedent_all"] or not dsl["requirement_all"]:
        raise ValueError("antecedent and requirement must be non-empty")
    for predicate in dsl["antecedent_all"] + dsl["requirement_all"] + dsl["exception_all"]:
        if predicate.get("operator") not in OPERATORS:
            raise ValueError(f"invalid predicate: {predicate}")


def evaluate_dsl(record: dict, dsl: dict) -> str:
    """Return one of four mutually exclusive semantic statuses.

    Exception has priority only after the antecedent applies.  This prevents an
    exception token elsewhere in a row from making an irrelevant request look
    active.
    """

    validate_dsl(dsl)
    if not evaluate_all(record, dsl["antecedent_all"]):
        return "not_applicable"
    if dsl["exception_all"] and evaluate_all(record, dsl["exception_all"]):
        return "excepted"
    if evaluate_all(record, dsl["requirement_all"]):
        return "satisfied"
    return "violated"


def action_for_status(status: str) -> str:
    if status not in STATUSES:
        raise ValueError(status)
    return "repair_candidate" if status == "violated" else "no_action"


def make_predicate(
    column: str,
    operator: str,
    *,
    string_value: str | None = None,
    number_value: float | None = None,
    string_list: list[str] | None = None,
) -> dict:
    if string_list is not None:
        value_type = "string_list"
    elif number_value is not None:
        value_type = "number"
    else:
        value_type = "string"
    return {
        "column": column,
        "operator": operator,
        "value_type": value_type,
        "string_value": string_value,
        "number_value": number_value,
        "string_list": string_list or [],
    }


def make_repair(column: str, *, string_value: str | None = None, number_value: float | None = None) -> dict:
    return {
        "column": column,
        "value_type": "number" if number_value is not None else "string",
        "string_value": string_value,
        "number_value": number_value,
    }
