#!/usr/bin/env python3
"""Audit record-level LLM critiques on synthetic TabDiff previews only."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = {
    "age": "integer age in years",
    "workclass": "employment sector category",
    "fnlwgt": "survey final weight",
    "education": "named highest education category",
    "education.num": "ordinal numeric code for education",
    "marital.status": "marital status category",
    "occupation": "occupation category",
    "relationship": "household relationship category",
    "race": "race category",
    "sex": "sex category",
    "capital.gain": "non-negative annual capital gain",
    "capital.loss": "non-negative annual capital loss",
    "hours.per.week": "positive weekly working hours",
    "native.country": "country category",
    "income": "binary income band",
}


STATIC_VALIDATED_RELATIONS = {
    "education_to_education.num": {
        "Preschool": 1,
        "1st-4th": 2,
        "5th-6th": 3,
        "7th-8th": 4,
        "9th": 5,
        "10th": 6,
        "11th": 7,
        "12th": 8,
        "HS-grad": 9,
        "Some-college": 10,
        "Assoc-voc": 11,
        "Assoc-acdm": 12,
        "Bachelors": 13,
        "Masters": 14,
        "Prof-school": 15,
        "Doctorate": 16,
    },
    "conditional_relations": [
        "relationship == Husband implies sex == Male and marital.status == Married-civ-spouse",
        "relationship == Wife implies sex == Female and marital.status == Married-civ-spouse",
    ],
    "range_relations": ["capital.gain >= 0", "capital.loss >= 0", "hours.per.week >= 1"],
}


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "record_is_valid": {"type": "boolean"},
        "critiques": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "violated_relation": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "applicability_condition": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "suggested_change": {"type": "string"},
                },
                "required": [
                    "violated_relation",
                    "columns",
                    "applicability_condition",
                    "confidence",
                    "suggested_change",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["record_is_valid", "critiques"],
    "additionalProperties": False,
}


def load_env(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def api_json(url: str, key: str, payload=None, timeout=120):
    headers = {"Authorization": f"Bearer {key}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, headers=headers, data=data, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def choose_model(key: str, requested: str | None) -> tuple[str, list[str]]:
    listing = api_json("https://api.openai.com/v1/models", key)
    available = sorted(item["id"] for item in listing.get("data", []))

    def usable(model: str) -> bool:
        try:
            api_json(
                "https://api.openai.com/v1/responses",
                key,
                {"model": model, "input": "Reply OK", "max_output_tokens": 20, "store": False},
            )
            return True
        except urllib.error.HTTPError:
            return False

    if requested:
        if requested not in available:
            raise RuntimeError(f"Requested model {requested!r} is not available to this API project")
        # An explicit model has already been selected by the caller. Avoid a
        # separate probe request here: transient rate limits on that probe can
        # otherwise abort an audit even though the actual calls are retried.
        return requested, available
    for candidate in ("gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini"):
        if candidate in available and usable(candidate):
            return candidate, available
    mini = [model for model in available if model.startswith("gpt-") and "mini" in model and "audio" not in model and "realtime" not in model]
    for candidate in reversed(mini):
        if usable(candidate):
            return candidate, available
    raise RuntimeError("No small text-capable GPT model was found; pass --model explicitly")


def call_critic(key: str, model: str, case: dict, include_static_rules: bool) -> tuple[dict, dict, float]:
    input_payload = {
        "schema": SCHEMA,
        "diffusion_t": case["t"],
        "synthetic_preview_record": case["record"],
    }
    if include_static_rules:
        input_payload["static_validated_relations"] = STATIC_VALIDATED_RELATIONS
    instructions = (
        "You are auditing one synthetic tabular preview from a diffusion trajectory. "
        "Use only the supplied schema and record. Identify concrete semantic or logical inconsistencies, "
        "not distributional rarity. Do not invent a violation when the record can plausibly be valid. "
        "Each suggested change must be typed and name the column to change. "
        "Ignore leading whitespace and harmless fractional preview residuals; never replace a rare demographic "
        "value merely because another value is more common. If static_validated_relations are supplied, check "
        "those exact relations and do not invent additional demographic correlations."
    )
    request = {
        "model": model,
        "instructions": instructions,
        "input": json.dumps(input_payload, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "tabular_record_critique",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "max_output_tokens": 800,
        "store": False,
    }
    start = time.perf_counter()
    response = api_json("https://api.openai.com/v1/responses", key, request)
    elapsed = time.perf_counter() - start
    output_text = response.get("output_text")
    if not output_text:
        chunks = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    chunks.append(content.get("text", ""))
        output_text = "".join(chunks)
    parsed = json.loads(output_text)
    return parsed, response.get("usage", {}), elapsed


def canonical_predictions(critique: dict) -> tuple[set[str], int, int]:
    predicted = set()
    actionable = 0
    unknown_columns = 0
    for item in critique.get("critiques", []):
        columns = {str(c) for c in item.get("columns", [])}
        text = (item.get("violated_relation", "") + " " + item.get("suggested_change", "")).lower()
        if columns and columns.issubset(SCHEMA) and item.get("suggested_change", "").strip():
            actionable += 1
        unknown_columns += len(columns - set(SCHEMA))
        if {"education", "education.num"}.issubset(columns) or ("education" in text and "ordinal" in text):
            predicted.add("education_fd")
        if {"relationship", "sex"}.issubset(columns):
            if "wife" in text or "female" in text:
                predicted.add("wife_female")
            if "husband" in text or "male" in text:
                predicted.add("husband_male")
            if "wife" not in text and "husband" not in text:
                predicted.update(["wife_female", "husband_male"])
        if {"relationship", "marital.status"}.issubset(columns):
            predicted.add("spouse_marital")
        if "capital.gain" in columns and ("negative" in text or "non-negative" in text or "nonnegative" in text):
            predicted.add("capital_gain_nonnegative")
        if "capital.loss" in columns and ("negative" in text or "non-negative" in text or "nonnegative" in text):
            predicted.add("capital_loss_nonnegative")
        if "hours.per.week" in columns and ("positive" in text or "zero" in text or "negative" in text):
            predicted.add("hours_positive")
    return predicted, actionable, unknown_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-cases", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--include-static-rules", action="store_true")
    parser.add_argument("--inter-call-delay", type=float, default=0.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    model, available = choose_model(key, args.model)

    all_cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    by_t = defaultdict(list)
    for case in all_cases:
        by_t[round(float(case["t"]), 3)].append(case)
    selected = []
    timestep_keys = sorted(by_t)
    target_times = (0.7, 0.4, 0.1)
    per_t = max(1, int(np.ceil(args.max_cases / len(target_times))))
    for target in target_times:
        t_key = min(timestep_keys, key=lambda value: abs(value - target))
        selected.extend(by_t[t_key][:per_t])
    selected = selected[: args.max_cases]

    rows, raw_rows = [], []
    for case in selected:
        gold = set(case.get("verifier_violations", []))
        for repeat in range(args.repeats):
            error = None
            parsed, usage, elapsed = {}, {}, float("nan")
            for attempt in range(3):
                try:
                    parsed, usage, elapsed = call_critic(key, model, case, args.include_static_rules)
                    error = None
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < 2:
                        time.sleep(2**attempt)
            predicted, actionable, unknown = canonical_predictions(parsed)
            tp = len(predicted & gold)
            fp = len(predicted - gold)
            fn = len(gold - predicted)
            row = {
                "case_id": case["case_id"],
                "t": case["t"],
                "repeat": repeat,
                "model": model,
                "condition": "static_validated_relations" if args.include_static_rules else "schema_only",
                "gold": "|".join(sorted(gold)),
                "predicted": "|".join(sorted(predicted)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "critique_count": len(parsed.get("critiques", [])),
                "actionable_count": actionable,
                "unknown_column_count": unknown,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "runtime_seconds": elapsed,
                "error": error,
            }
            rows.append(row)
            raw_rows.append({"case_id": case["case_id"], "t": case["t"], "repeat": repeat, "critique": parsed, "usage": usage, "error": error})
            print(json.dumps(row), flush=True)
            if args.inter_call_delay > 0:
                time.sleep(args.inter_call_delay)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "llm_feedback_cases.csv", index=False)
    with (args.output / "llm_feedback_raw.jsonl").open("w") as handle:
        for item in raw_rows:
            handle.write(json.dumps(item) + "\n")

    valid = frame[frame["error"].isna()]
    disagreements = []
    for _, group in valid.groupby("case_id"):
        sets = [set(value.split("|")) if value else set() for value in group["predicted"]]
        if len(sets) >= 2:
            union = sets[0] | sets[1]
            disagreements.append(1 - (len(sets[0] & sets[1]) / len(union) if union else 1.0))
    tp, fp, fn = valid[["tp", "fp", "fn"]].sum() if len(valid) else (0, 0, 0)
    summary = {
        "status": "executed" if len(valid) else "failed",
        "model": model,
        "condition": "static_validated_relations" if args.include_static_rules else "schema_only",
        "available_model_count": len(available),
        "cases": len(selected),
        "calls_attempted": len(frame),
        "calls_succeeded": len(valid),
        "micro_precision_strict_verifier": float(tp / (tp + fp)) if tp + fp else None,
        "micro_recall_strict_verifier": float(tp / (tp + fn)) if tp + fn else None,
        "mean_repeat_disagreement_jaccard": float(np.mean(disagreements)) if disagreements else None,
        "actionable_fraction": float((valid["actionable_count"] > 0).mean()) if len(valid) else None,
        "input_tokens": int(valid["input_tokens"].sum()) if len(valid) else 0,
        "output_tokens": int(valid["output_tokens"].sum()) if len(valid) else 0,
        "total_tokens": int(valid["total_tokens"].sum()) if len(valid) else 0,
        "runtime_seconds": float(valid["runtime_seconds"].sum()) if len(valid) else 0,
        "privacy_boundary": "Only schema metadata and synthetic preview records were sent; no real row was sent.",
    }
    (args.output / "llm_feedback_summary.json").write_text(json.dumps(summary, indent=2))
    if len(valid):
        valid.groupby("t").agg(
            calls=("case_id", "size"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tokens=("total_tokens", "sum"),
            runtime_seconds=("runtime_seconds", "sum"),
        ).to_csv(args.output / "llm_feedback_by_timestep.csv")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
