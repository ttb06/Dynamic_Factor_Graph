#!/usr/bin/env python3
"""Compare direct FM judgement with NL -> typed DSL -> local execution.

Privacy boundary: the API receives only synthetic TabDiff preview records and
natural-language generation requests.  Oracle DSLs, gold statuses, internal
case IDs, pair IDs, and design labels never enter an API payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import urllib.error
from pathlib import Path

import pandas as pd

from llm_feedback_audit import SCHEMA, load_env
from request_condition_common import action_for_status, evaluate_dsl, validate_dsl
from semantic_llm_sweep import api_json, output_text


STATUSES = ["violated", "satisfied", "not_applicable", "excepted"]
ACTIONS = ["repair_candidate", "no_action"]
PREDICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "column": {"type": "string"},
        "operator": {"type": "string", "enum": ["eq", "neq", "in", "not_in", "gte", "lte", "gt", "lt"]},
        "value_type": {"type": "string", "enum": ["string", "number", "string_list"]},
        "string_value": {"type": "string"},
        "number_value": {"type": "number"},
        "string_list": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["column", "operator", "value_type", "string_value", "number_value", "string_list"],
    "additionalProperties": False,
}
REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "column": {"type": "string"},
        "value_type": {"type": "string", "enum": ["string", "number"]},
        "string_value": {"type": "string"},
        "number_value": {"type": "number"},
    },
    "required": ["column", "value_type", "string_value", "number_value"],
    "additionalProperties": False,
}
DSL_SCHEMA = {
    "type": "object",
    "properties": {
        "antecedent_all": {"type": "array", "items": PREDICATE_SCHEMA},
        "requirement_all": {"type": "array", "items": PREDICATE_SCHEMA},
        "exception_all": {"type": "array", "items": PREDICATE_SCHEMA},
        "repair": REPAIR_SCHEMA,
    },
    "required": ["antecedent_all", "requirement_all", "exception_all", "repair"],
    "additionalProperties": False,
}
DIRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "status": {"type": "string", "enum": STATUSES},
                    "action": {"type": "string", "enum": ACTIONS},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["case_id", "status", "action", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}
COMPILE_SCHEMA = {
    "type": "object",
    "properties": {
        "compiled_requests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"request_id": {"type": "string"}, "dsl": DSL_SCHEMA},
                "required": ["request_id", "dsl"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["compiled_requests"],
    "additionalProperties": False,
}


def opaque_id(seed: int, kind: str, ordinal: int, private_id: str) -> str:
    digest = hashlib.sha256(f"{seed}:{kind}:{ordinal}:{private_id}".encode()).hexdigest()[:18]
    return f"x_{digest}"


def request_api(key: str, model: str, instructions: str, payload: dict, schema_name: str, schema: dict, max_tokens: int):
    body = {
        "model": model,
        "instructions": instructions,
        "input": json.dumps(payload, ensure_ascii=False),
        "text": {"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}},
        "max_output_tokens": max_tokens,
        "store": False,
    }
    started = time.perf_counter()
    response = api_json("https://api.openai.com/v1/responses", key, body, timeout=240)
    parsed = json.loads(output_text(response))
    return parsed, response.get("usage", {}), time.perf_counter() - started


def with_retries(callable_):
    error = None
    for attempt in range(5):
        try:
            parsed, usage, elapsed = callable_()
            return parsed, usage, elapsed, None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            error = f"HTTPError {exc.code}: {body}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:  # malformed model output is an experimental failure, not a crash
            error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(30.0, 2 ** attempt + random.random()))
    return {}, {}, math.nan, error


def direct_instructions() -> str:
    return (
        "Judge each synthetic record only against its referenced generation request. The request is authoritative; "
        "do not replace it with Adult-dataset correlations or demographic typicality. Use this exact precedence: "
        "(1) if the IF antecedent is false, status=not_applicable; (2) if an explicit exception applies, "
        "status=excepted; (3) if the requirement holds, status=satisfied; otherwise status=violated. "
        "Use repair_candidate only for violated and no_action for every other status. Ignore leading whitespace. "
        "Return exactly one decision for every opaque case_id."
    )


def compile_instructions(strategy: str = "basic") -> str:
    base = (
        "Compile each natural-language tabular generation request into the supplied typed DSL. Do not evaluate any "
        "row: no rows are supplied. antecedent_all encodes the IF clause; requirement_all encodes what must hold; "
        "exception_all encodes an explicit exception (empty if none); repair encodes the requested deterministic "
        "change for a true violation. Conjunction is used within each list. For an unused scalar field in a typed "
        "predicate, emit an empty string, 0, or empty list as appropriate. Preserve exact column/category spelling. "
        "If categorical_values metadata is supplied, every categorical literal must be copied exactly from it; "
        "map descriptive wording in the request to the corresponding canonical value rather than inventing a synonym. "
        "Return exactly one compiled item for every opaque request_id."
    )
    if strategy == "basic":
        return base
    if strategy == "decomposed":
        return base + (
            " Before emitting the DSL, semantically decompose every policy into four roles: SCOPE (when the row is "
            "in scope), EXCEPTION (an in-scope row that must not be treated as a violation), REQUIREMENT (what an "
            "in-scope non-excepted row must satisfy), and REPAIR. Map these roles by meaning, not sentence order. "
            "In particular, an exception mentioned before the scope still belongs in exception_all. In a formula "
            "of the form (A) AND NOT (E) IMPLIES (R), map A to antecedent_all, E to exception_all, and R to "
            "requirement_all. Do not copy an exception predicate into the antecedent."
        )
    raise ValueError(f"unknown compile strategy: {strategy}")


def direct_payload(batch: list[dict], request_by_id: dict[str, dict], opaque_requests: dict[str, str]) -> dict:
    active = sorted({case["request_id"] for case in batch})
    return {
        "schema": SCHEMA,
        "generation_requests": [
            {"request_id": opaque_requests[request_id], "request_text": request_by_id[request_id]["text"]}
            for request_id in active
        ],
        "synthetic_cases": [
            {
                "case_id": case["api_case_id"],
                "request_id": opaque_requests[case["request_id"]],
                "record": case["record"],
            }
            for case in batch
        ],
    }


def normalize_predicate(predicate: dict) -> tuple:
    value_type = predicate["value_type"]
    if value_type == "string":
        value = str(predicate.get("string_value", "")).strip()
    elif value_type == "number":
        value = round(float(predicate.get("number_value", 0)), 8)
    else:
        value = tuple(str(item).strip() for item in predicate.get("string_list", []))
    return predicate["column"], predicate["operator"], value_type, value


def normalized_dsl(dsl: dict) -> tuple:
    repair = dsl["repair"]
    repair_value = (
        round(float(repair.get("number_value", 0)), 8)
        if repair["value_type"] == "number"
        else str(repair.get("string_value", "")).strip()
    )
    return (
        tuple(normalize_predicate(item) for item in dsl["antecedent_all"]),
        tuple(normalize_predicate(item) for item in dsl["requirement_all"]),
        tuple(normalize_predicate(item) for item in dsl["exception_all"]),
        (repair["column"], repair["value_type"], repair_value),
    )


def pair_metrics(frame: pd.DataFrame) -> dict:
    pairs = frame[frame["pair_id"].fillna("").ne("")]
    if pairs.empty:
        return {"request_swap_pairs": 0, "request_swap_pair_accuracy": None, "request_swap_flip_sensitivity": None}
    grouped = list(pairs.groupby("pair_id"))
    exact = []
    sensitivity = []
    for _, group in grouped:
        exact.append(bool((group["gold_status"] == group["pred_status"]).all() and len(group) == 2))
        sensitivity.append(group["pred_status"].nunique() == 2 and set(group["pred_status"]) == {"satisfied", "violated"})
    return {
        "request_swap_pairs": len(grouped),
        "request_swap_pair_accuracy": float(sum(exact) / len(exact)),
        "request_swap_flip_sensitivity": float(sum(sensitivity) / len(sensitivity)),
    }


def decision_summary(frame: pd.DataFrame, *, mode: str, model: str, usage: dict, latency: float, extra: dict | None = None) -> dict:
    available = frame["pred_status"].ne("missing")
    evaluated = frame[available]
    summary = {
        "status": "executed" if not evaluated.empty else "failed",
        "mode": mode,
        "model": model,
        "cases": len(frame),
        "four_way_status_accuracy": float((evaluated["gold_status"] == evaluated["pred_status"]).mean()) if len(evaluated) else None,
        "action_accuracy": float((evaluated["gold_action"] == evaluated["pred_action"]).mean()) if len(evaluated) else None,
        "exception_false_action_rate": float(frame.loc[frame.gold_status.eq("excepted"), "pred_action"].eq("repair_candidate").mean()),
        "not_applicable_false_action_rate": float(frame.loc[frame.gold_status.eq("not_applicable"), "pred_action"].eq("repair_candidate").mean()),
        "satisfied_false_action_rate": float(frame.loc[frame.gold_status.eq("satisfied"), "pred_action"].eq("repair_candidate").mean()),
        "missing_rate": float(frame["pred_status"].eq("missing").mean()),
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "latency_seconds": latency,
        "privacy_boundary": "Only synthetic preview rows plus request text were sent; no real rows.",
        "leakage_control": "Opaque randomized API IDs; oracle DSL, pair IDs, design labels, and gold stay local.",
        **pair_metrics(frame),
    }
    if extra:
        summary.update(extra)
    return summary


def run_direct(args, key: str, cases: list[dict], requests: list[dict], opaque_requests: dict[str, str]) -> None:
    request_by_id = {request["request_id"]: request for request in requests}
    rows, raw = [], []
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    latency = 0.0
    for start in range(0, len(cases), args.batch_size):
        batch = cases[start:start + args.batch_size]
        parsed, usage, elapsed, error = with_retries(
            lambda: request_api(
                key, args.model, direct_instructions(), direct_payload(batch, request_by_id, opaque_requests),
                "request_condition_decisions", DIRECT_SCHEMA, max(1200, 160 * len(batch)),
            )
        )
        returned = {item.get("case_id"): item for item in parsed.get("decisions", [])} if error is None else {}
        if error is None:
            latency += elapsed
            for name in totals:
                totals[name] += int(usage.get(name, 0) or 0)
        raw.append({"batch_start": start, "response": parsed, "usage": usage, "latency": elapsed, "error": error})
        for case in batch:
            decision = returned.get(case["api_case_id"], {})
            rows.append(
                {
                    "case_id": case["case_id"],
                    "api_case_id": case["api_case_id"],
                    "request_id": case["request_id"],
                    "pair_id": case.get("pair_id"),
                    "t": case["t"],
                    "design": case["design"],
                    "gold_status": case["gold_status"],
                    "pred_status": decision.get("status", "missing"),
                    "gold_action": case["gold_action"],
                    "pred_action": decision.get("action", "missing"),
                    "confidence": decision.get("confidence"),
                    "error": error or "",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "decisions.csv", index=False)
    with (args.output / "responses.jsonl").open("w") as handle:
        for item in raw:
            handle.write(json.dumps(item) + "\n")
    summary = decision_summary(frame, mode="direct", model=args.model, usage=totals, latency=latency)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def run_compile(args, key: str, cases: list[dict], requests: list[dict], opaque_requests: dict[str, str]) -> None:
    payload = {
        "schema": SCHEMA,
        "generation_requests": [
            {"request_id": opaque_requests[request["request_id"]], "request_text": request["text"]}
            for request in requests
        ],
    }
    if args.categorical_vocabulary:
        payload["categorical_values"] = json.loads(args.categorical_vocabulary.read_text())
    parsed, usage, latency, error = with_retries(
        lambda: request_api(
            key, args.model, compile_instructions(args.compile_strategy), payload,
            "compiled_tabular_requests", COMPILE_SCHEMA, max(1800, 450 * len(requests)),
        )
    )
    returned = {item.get("request_id"): item.get("dsl") for item in parsed.get("compiled_requests", [])} if error is None else {}
    request_by_id = {request["request_id"]: request for request in requests}
    compile_rows, predicted_dsls = [], {}
    for request in requests:
        api_id = opaque_requests[request["request_id"]]
        dsl = returned.get(api_id)
        valid, execution_error = False, ""
        if dsl is not None:
            try:
                validate_dsl(dsl)
                valid = True
                predicted_dsls[request["request_id"]] = dsl
            except Exception as exc:
                execution_error = f"{type(exc).__name__}: {exc}"
        compile_rows.append(
            {
                "request_id": request["request_id"],
                "api_request_id": api_id,
                "compile_valid": valid,
                "structural_exact_match": bool(valid and normalized_dsl(dsl) == normalized_dsl(request["oracle_dsl"])),
                "compile_error": execution_error or error or "",
                "predicted_dsl": json.dumps(dsl) if dsl is not None else "",
            }
        )
    pd.DataFrame(compile_rows).to_csv(args.output / "compiled_requests.csv", index=False)
    # The response contains opaque IDs and predicted DSL only; write local mapping separately for auditability.
    (args.output / "response.json").write_text(json.dumps({"response": parsed, "usage": usage, "latency": latency, "error": error}, indent=2))

    rows = []
    for case in cases:
        dsl = predicted_dsls.get(case["request_id"])
        execution_error = ""
        if dsl is None:
            status = "missing"
        else:
            try:
                status = evaluate_dsl(case["record"], dsl)
            except Exception as exc:
                status = "missing"
                execution_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "case_id": case["case_id"],
                "request_id": case["request_id"],
                "pair_id": case.get("pair_id"),
                "t": case["t"],
                "design": case["design"],
                "gold_status": case["gold_status"],
                "pred_status": status,
                "gold_action": case["gold_action"],
                "pred_action": action_for_status(status) if status != "missing" else "missing",
                "execution_error": execution_error,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "decisions.csv", index=False)
    compile_frame = pd.DataFrame(compile_rows)
    extra = {
        "requests": len(requests),
        "compile_valid_rate": float(compile_frame.compile_valid.mean()),
        "dsl_structural_exact_match_rate": float(compile_frame.structural_exact_match.mean()),
        "execution_equivalence": float((frame.gold_status == frame.pred_status).mean()),
        "compile_error": error,
        "compile_strategy": args.compile_strategy,
    }
    summary = decision_summary(frame, mode="compile_execute", model=args.model, usage=usage, latency=0.0 if math.isnan(latency) else latency, extra=extra)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["direct", "compile"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--categorical-vocabulary", type=Path)
    parser.add_argument("--schema-file", type=Path, help="Optional dataset-specific schema JSON.")
    parser.add_argument("--compile-strategy", choices=["basic", "decomposed"], default="basic")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    global SCHEMA
    if args.schema_file:
        SCHEMA = json.loads(args.schema_file.read_text())
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    requests = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    if any(case.get("privacy") != "synthetic_preview_only" for case in cases):
        raise RuntimeError("refusing API call: case is not marked synthetic_preview_only")

    rng = random.Random(args.seed)
    rng.shuffle(cases)
    rng.shuffle(requests)
    for ordinal, case in enumerate(cases):
        case["api_case_id"] = opaque_id(args.seed, "case", ordinal, case["case_id"])
    opaque_requests = {
        request["request_id"]: opaque_id(args.seed, "request", ordinal, request["request_id"])
        for ordinal, request in enumerate(requests)
    }
    # The map is deliberately local and lets an auditor verify that private IDs were not sent.
    (args.output / "local_id_map.json").write_text(
        json.dumps({"requests": opaque_requests, "cases": {case["case_id"]: case["api_case_id"] for case in cases}}, indent=2)
    )
    if args.mode == "direct":
        run_direct(args, key, cases, requests, opaque_requests)
    else:
        run_compile(args, key, cases, requests, opaque_requests)


if __name__ == "__main__":
    main()
