#!/usr/bin/env python3
"""Batched structured LLM audit over paired synthetic trajectory cases."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from llm_feedback_audit import SCHEMA, STATIC_VALIDATED_RELATIONS, load_env


RELATIONS = [
    "education_fd",
    "numeric_domain",
    "spouse_consistency",
    "age_education_plausibility",
    "missingness_coupling",
    "none",
    "other",
]
STATUS = ["violated", "satisfied", "not_applicable", "abstain"]
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "status": {"type": "string", "enum": STATUS},
                    "relation": {"type": "string", "enum": RELATIONS},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rationale_code": {
                        "type": "string",
                        "enum": [
                            "direct_rule",
                            "world_knowledge",
                            "temporal_discontinuity",
                            "rarity_only",
                            "insufficient_evidence",
                        ],
                    },
                    "action": {
                        "type": "string",
                        "enum": [
                            "rollback_candidate",
                            "project_numeric",
                            "project_categorical",
                            "no_action",
                        ],
                    },
                },
                "required": [
                    "case_id",
                    "status",
                    "relation",
                    "confidence",
                    "columns",
                    "rationale_code",
                    "action",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


SEMANTIC_RUBRIC = [
    "education and education.num should describe the same attained level",
    "gain/loss counts must not be negative and weekly hours must be positive",
    "household spouse labels must agree with the dataset's sex and marital-status coding",
    "very advanced attained education at an implausibly young age is inconsistent",
    "the source uses '?' as a coupled missingness sentinel for workclass and occupation",
]

RELATION_RUBRIC = {
    "education_fd": "education and education.num must denote the same attained education level",
    "numeric_domain": "capital gain/loss cannot be negative and hours per week must be positive",
    "spouse_consistency": "household spouse labels must agree with sex and marital-status coding",
    "age_education_plausibility": "very advanced attained education is impossible at a very young age",
    "missingness_coupling": "the '?' sentinel for workclass and occupation must occur together",
}

STATIC_RULES = {
    **STATIC_VALIDATED_RELATIONS,
    "age_education": ["Doctorate or Prof-school requires age >= 20", "Masters requires age >= 18"],
    "missingness": "workclass == '?' iff occupation == '?'",
}

FEW_SHOT = [
    {
        "record": {"age": 14, "education": "Doctorate", "education.num": 16},
        "decision": {"status": "violated", "relation": "age_education_plausibility", "action": "rollback_candidate"},
    },
    {
        "record": {"age": 45, "relationship": "Own-child", "native.country": "France"},
        "decision": {"status": "satisfied", "relation": "none", "action": "no_action"},
    },
    {
        "record": {"workclass": "?", "occupation": "Exec-managerial"},
        "decision": {"status": "violated", "relation": "missingness_coupling", "action": "rollback_candidate"},
    },
]


def api_json(url: str, key: str, payload=None, timeout=180):
    headers = {"Authorization": f"Bearer {key}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        headers=headers,
        data=data,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def output_text(response: dict) -> str:
    # UnoRouter may wrap an OpenAI-compatible inference payload in ``data``.
    # Accept that form without changing callers or logging credentials.
    if isinstance(response.get("data"), dict):
        response = response["data"]
    if response.get("output_text"):
        return response["output_text"]
    chunks = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    if chunks:
        return "".join(chunks)
    # Some routed providers normalize a Responses request to the Chat
    # Completions response shape.  Structured JSON is then returned as message
    # content (occasionally as content blocks).
    for choice in response.get("choices", []):
        content = choice.get("message", {}).get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return "".join(chunks)


def condition_instructions(condition: str) -> str:
    base = (
        "Audit synthetic tabular diffusion previews. Return exactly one decision per case_id. "
        "A valid or merely rare record must use status=satisfied, relation=none, action=no_action. "
        "Use violated only for a concrete inconsistency; never infer demographic typicality. "
        "Use abstain when evidence is insufficient. Leading whitespace is harmless."
    )
    additions = {
        "schema_only": " Use only the schema and current record.",
        "typed_contract": (
            " Enforce the status contract strictly: a checked-and-satisfied or non-applicable relation must not "
            "appear as violated. An action is allowed only when status=violated."
        ),
        "semantic_rubric": " Apply the supplied broad semantic rubric conservatively.",
        "static_rules": " Check only the supplied executable rules and do not invent other dependencies.",
        "fewshot": " Follow the supplied fabricated examples and preserve rare-but-valid controls.",
        "temporal_pair": (
            " Compare previous_record with current record. Temporal change is supporting evidence, not proof; "
            "still require a semantic inconsistency."
        ),
        "temporal_delta": (
            " Use changed_columns_from_previous as supporting evidence, not proof; still require a semantic inconsistency."
        ),
        "cost_sensitive_abstain": (
            " False positive repair costs five times more than a missed issue. Abstain unless confidence is at least 0.9."
        ),
        "focused_label": (
            " Verify only candidate_relation_to_verify. If violated, relation must equal that candidate; "
            "otherwise relation=none. Do not diagnose any other relation."
        ),
        "focused_rubric": (
            " Verify only candidate_relation_to_verify using its relation_description. If violated, relation must "
            "equal that candidate; otherwise relation=none. Do not diagnose any other relation."
        ),
        "focused_temporal": (
            " Verify only candidate_relation_to_verify using its relation_description and previous_record. "
            "Temporal change is supporting evidence, not proof. If violated, relation must equal the candidate; "
            "otherwise relation=none. Do not diagnose any other relation."
        ),
    }
    if condition not in additions:
        raise ValueError(condition)
    return base + additions[condition]


def make_payload(cases: list[dict], condition: str) -> dict:
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "diffusion_t": case["t"], "record": case["record"]}
        if condition == "temporal_pair":
            row["previous_record"] = case.get("previous_record")
        if condition == "temporal_delta":
            row["changed_columns_from_previous"] = case.get("changed_columns_from_previous", [])
        if condition in {"focused_label", "focused_rubric", "focused_temporal"}:
            relation = case["intended_relation"]
            row["candidate_relation_to_verify"] = relation
            if condition in {"focused_rubric", "focused_temporal"}:
                row["relation_description"] = RELATION_RUBRIC.get(
                    relation,
                    "No specific candidate relation is active; do not invent a violation.",
                )
            if condition == "focused_temporal":
                row["previous_record"] = case.get("previous_record")
        rows.append(row)
    payload = {"schema": SCHEMA, "synthetic_cases": rows}
    if condition == "semantic_rubric":
        payload["semantic_rubric"] = SEMANTIC_RUBRIC
    elif condition == "static_rules":
        payload["static_validated_rules"] = STATIC_RULES
    elif condition == "fewshot":
        payload["fabricated_examples"] = FEW_SHOT
    return payload


def call_batch(key: str, model: str, cases: list[dict], condition: str) -> tuple[dict, dict, float]:
    request = {
        "model": model,
        "instructions": condition_instructions(condition),
        "input": json.dumps(make_payload(cases, condition), ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "semantic_tabular_decisions",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "max_output_tokens": max(1200, 220 * len(cases)),
        "store": False,
    }
    start = time.perf_counter()
    response = api_json("https://api.openai.com/v1/responses", key, request)
    elapsed = time.perf_counter() - start
    return json.loads(output_text(response)), response.get("usage", {}), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=[
        "schema_only",
        "typed_contract",
        "semantic_rubric",
        "static_rules",
        "fewshot",
        "temporal_pair",
        "temporal_delta",
        "cost_sensitive_abstain",
        "focused_label",
        "focused_rubric",
        "focused_temporal",
    ], required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--inter-call-delay", type=float, default=3.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")

    all_cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.max_cases > 0:
        all_cases = all_cases[: args.max_cases]
    by_id = {case["case_id"]: case for case in all_cases}
    raw_handle = (args.output / "responses.jsonl").open("w")
    detail_rows = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    successful_calls = 0
    failed_calls = 0
    total_latency = 0.0
    available_models = []
    try:
        listing = api_json("https://api.openai.com/v1/models", key)
        available_models = sorted(item["id"] for item in listing.get("data", []))
    except Exception as exc:  # model call below remains the actual access check
        available_models = [f"listing_error:{type(exc).__name__}"]

    for repeat in range(args.repeats):
        for start in range(0, len(all_cases), args.batch_size):
            batch = all_cases[start : start + args.batch_size]
            parsed, usage, latency, error = {}, {}, math.nan, None
            for attempt in range(5):
                try:
                    parsed, usage, latency = call_batch(key, args.model, batch, args.condition)
                    error = None
                    break
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")[:500]
                    error = f"HTTPError {exc.code}: {body}"
                    if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                        break
                    retry_after = float(exc.headers.get("Retry-After", 0) or 0)
                    time.sleep(min(30.0, max(retry_after, 2**attempt + random.random())))
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    time.sleep(min(30.0, 2**attempt + random.random()))

            expected_ids = {case["case_id"] for case in batch}
            if error is None:
                successful_calls += 1
                total_latency += latency
                for key_name in usage_totals:
                    usage_totals[key_name] += int(usage.get(key_name, 0) or 0)
                decisions = parsed.get("decisions", [])
            else:
                failed_calls += 1
                decisions = []
            returned = {item.get("case_id"): item for item in decisions}
            raw_handle.write(
                json.dumps(
                    {
                        "repeat": repeat,
                        "batch_start": start,
                        "expected_case_ids": sorted(expected_ids),
                        "parsed": parsed,
                        "usage": usage,
                        "latency_seconds": latency,
                        "error": error,
                    }
                )
                + "\n"
            )
            raw_handle.flush()
            for case_id in sorted(expected_ids):
                case = by_id[case_id]
                decision = returned.get(case_id, {})
                status = decision.get("status", "missing")
                relation = decision.get("relation", "none")
                predicted = relation if status == "violated" and relation not in {"none", "other"} else ""
                gold = set(case.get("gold_relations", []))
                detail_rows.append(
                    {
                        "case_id": case_id,
                        "semantic_key": case["semantic_key"],
                        "pair_id": case["pair_id"],
                        "t": case["t"],
                        "control_type": case["control_type"],
                        "intended_relation": case["intended_relation"],
                        "gold": "|".join(sorted(gold)),
                        "repeat": repeat,
                        "condition": args.condition,
                        "model": args.model,
                        "status": status,
                        "predicted": predicted,
                        "confidence": decision.get("confidence"),
                        "columns": "|".join(decision.get("columns", [])),
                        "rationale_code": decision.get("rationale_code"),
                        "action": decision.get("action"),
                        "missing_decision": case_id not in returned,
                        "batch_error": error,
                    }
                )
            if start + args.batch_size < len(all_cases) or repeat + 1 < args.repeats:
                time.sleep(args.inter_call_delay)
    raw_handle.close()
    details = pd.DataFrame(detail_rows)
    details.to_csv(args.output / "decisions.csv", index=False)

    tp = fp = fn = 0
    for row in detail_rows:
        gold = set(filter(None, row["gold"].split("|")))
        pred = {row["predicted"]} if row["predicted"] else set()
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    summary = {
        "status": "executed" if successful_calls else "api_failed",
        "condition": args.condition,
        "model": args.model,
        "cases": len(all_cases),
        "repeats": args.repeats,
        "successful_calls": successful_calls,
        "failed_calls": failed_calls,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "input_tokens": usage_totals["input_tokens"],
        "output_tokens": usage_totals["output_tokens"],
        "total_tokens": usage_totals["total_tokens"],
        "api_latency_seconds": total_latency,
        "available_model_count": len(available_models),
        "privacy_boundary": "Only schema, fabricated examples/rules, and synthetic preview records; no real row.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
