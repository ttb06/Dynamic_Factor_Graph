#!/usr/bin/env python3
"""Structured API audit of dynamic constraint applicability and satisfaction."""

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
from semantic_llm_sweep import api_json, output_text


CONSTRAINT_IDS = [
    "husband_sex", "wife_sex", "spouse_marital", "advanced_education_age",
    "masters_age", "workclass_missingness", "occupation_missingness", "none",
]
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"decisions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "applicability": {"type": "string", "enum": ["applicable", "not_applicable", "uncertain"]},
            "status": {"type": "string", "enum": ["violated", "satisfied", "not_applicable", "abstain"]},
            "relation": {"type": "string", "enum": CONSTRAINT_IDS},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "action": {"type": "string", "enum": ["rollback_candidate", "no_action"]},
        },
        "required": ["case_id", "applicability", "status", "relation", "confidence", "action"],
        "additionalProperties": False,
    }}},
    "required": ["decisions"], "additionalProperties": False,
}


def instructions(condition: str) -> str:
    base = (
        "Audit synthetic tabular diffusion previews. First decide whether the target constraint's antecedent applies. "
        "If it does not, return applicability=not_applicable, status=not_applicable, relation=none, action=no_action. "
        "If it applies, separately decide whether the consequent is satisfied. Only a concrete violation permits "
        "action=rollback_candidate. Return exactly one decision per case_id and do not diagnose untargeted relations."
    )
    additions = {
        "id_only": " Infer the meaning of the target constraint only from its identifier and schema.",
        "natural_language": " Use the supplied natural-language constraint description.",
        "structured_predicate": " Use the supplied structured antecedent and consequent as the authoritative condition.",
        "temporal_nl": " Use the natural-language constraint and previous preview; temporal change is evidence, not truth.",
    }
    return base + additions[condition]


def payload(cases: list[dict], condition: str) -> dict:
    rows = []
    for case in cases:
        # Only this opaque identifier is sent to the model.  The private case_id
        # contains benchmark labels and must never cross the API boundary.
        row = {"case_id": case["api_case_id"], "diffusion_t": case["t"], "record": case["record"], "target_constraint_id": case["constraint_id"]}
        if condition in {"natural_language", "temporal_nl"}:
            row["constraint_description"] = case["constraint_description"]
        if condition == "structured_predicate":
            row["structured_predicate"] = case["structured_predicate"]
        if condition == "temporal_nl":
            row["previous_record"] = case.get("previous_record")
        rows.append(row)
    return {"schema": SCHEMA, "synthetic_cases": rows}


def call(key: str, model: str, cases: list[dict], condition: str):
    request = {
        "model": model,
        "instructions": instructions(condition),
        "input": json.dumps(payload(cases, condition), ensure_ascii=False),
        "text": {"format": {"type": "json_schema", "name": "dynamic_constraint_decisions", "strict": True, "schema": OUTPUT_SCHEMA}},
        "max_output_tokens": max(1200, 180 * len(cases)),
        "store": False,
    }
    started = time.perf_counter()
    response = api_json("https://api.openai.com/v1/responses", key, request)
    return json.loads(output_text(response)), response.get("usage", {}), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=["id_only", "natural_language", "structured_predicate", "temporal_nl"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    # Randomize before batching so clean/corrupt or satisfied/violated variants
    # are not systematically co-located.  Opaque IDs prevent label leakage.
    random.Random(args.seed).shuffle(cases)
    for index, case in enumerate(cases):
        digest = hashlib.sha256(f"{args.seed}:{index}:{case['case_id']}".encode()).hexdigest()[:16]
        case["api_case_id"] = f"r_{digest}"
    rows, raw = [], []
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    latency = 0.0
    for start in range(0, len(cases), args.batch_size):
        batch = cases[start:start + args.batch_size]
        parsed, usage, elapsed, error = {}, {}, math.nan, None
        for attempt in range(5):
            try:
                parsed, usage, elapsed = call(key, args.model, batch, args.condition)
                error = None; break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                error = f"HTTPError {exc.code}: {body}"
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}: break
                time.sleep(min(30.0, 2 ** attempt + random.random()))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(30.0, 2 ** attempt + random.random()))
        returned = {item.get("case_id"): item for item in parsed.get("decisions", [])} if error is None else {}
        if error is None:
            latency += elapsed
            for name in totals: totals[name] += int(usage.get(name, 0) or 0)
        raw.append({"batch_start": start, "parsed": parsed, "usage": usage, "latency": elapsed, "error": error})
        for case in batch:
            d = returned.get(case["api_case_id"], {})
            rows.append({
                "case_id": case["case_id"], "api_case_id": case["api_case_id"], "t": case["t"], "constraint_id": case["constraint_id"], "state": case["state"],
                "gold_applicability": case["gold_applicability"], "gold_status": case["gold_status"],
                "pred_applicability": d.get("applicability", "missing"), "pred_status": d.get("status", "missing"),
                "pred_relation": d.get("relation", "none"), "confidence": d.get("confidence"), "action": d.get("action", "missing"),
                "missing_decision": not bool(d), "batch_error": error or "", "condition": args.condition, "model": args.model,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "decisions.csv", index=False)
    with (args.output / "responses.jsonl").open("w") as handle:
        for item in raw: handle.write(json.dumps(item) + "\n")
    valid = ~frame["missing_decision"]
    violation_gold = frame["gold_status"].eq("violated")
    violation_pred = frame["pred_status"].eq("violated")
    tp = int((violation_gold & violation_pred).sum()); fp = int((~violation_gold & violation_pred).sum()); fn = int((violation_gold & ~violation_pred).sum())
    summary = {
        "status": "executed" if valid.any() else "failed", "condition": args.condition, "model": args.model, "cases": len(frame),
        "applicability_accuracy": float((frame.loc[valid, "gold_applicability"] == frame.loc[valid, "pred_applicability"]).mean()) if valid.any() else None,
        "status_accuracy": float((frame.loc[valid, "gold_status"] == frame.loc[valid, "pred_status"]).mean()) if valid.any() else None,
        "violation_precision": tp / (tp + fp) if tp + fp else None, "violation_recall": tp / (tp + fn) if tp + fn else None,
        "nonapplicable_action_rate": float(frame.loc[frame.gold_status.eq("not_applicable"), "action"].eq("rollback_candidate").mean()),
        "satisfied_action_rate": float(frame.loc[frame.gold_status.eq("satisfied"), "action"].eq("rollback_candidate").mean()),
        "missing_rate": float(frame.missing_decision.mean()), **totals, "latency_seconds": latency,
        "privacy_boundary": "Only synthetic preview rows and constraints were sent; no real rows.",
        "leakage_control": "Opaque randomized case IDs; benchmark labels remain local; variants shuffled before batching.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
