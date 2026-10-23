#!/usr/bin/env python3
"""Use an LLM as a semantic selector for train-discovered subgroup graph edges.

The API receives schema plus train aggregates only. Heldout rates, replication
labels, raw rows, and local constraint identifiers are withheld.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import urllib.error
from pathlib import Path

import pandas as pd

from llm_feedback_audit import load_env
from semantic_llm_sweep import api_json, output_text


ACTIONS = [
    "activate_in_group",
    "deactivate_in_group_as_exception",
    "soft_dependency_only",
    "audit_only",
    "reject_spurious",
]
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "action": {"type": "string", "enum": ACTIONS},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "semantic_rationale": {"type": "string"},
                },
                "required": ["candidate_id", "action", "confidence", "semantic_rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def instructions() -> str:
    return (
        "You are selecting typed subgroup-conditioned edges for a synthetic-tabular constraint graph. "
        "The supplied rates are empirical training aggregates, not proof of a semantic invariant. "
        "Return one decision for every candidate_id. Use activate_in_group only when the relation is "
        "semantically defensible as a stronger requirement inside the subgroup. Use "
        "deactivate_in_group_as_exception only when the subgroup plausibly explains why an otherwise useful "
        "edge should not be enforced there. Use soft_dependency_only for plausible distributional structure "
        "that must not become hard validity. Use audit_only for sensitive/demographic structure. Reject likely "
        "coincidence, redundant bookkeeping correlations, target leakage, or a subgroup that merely restates "
        "the edge. Never infer causality from confidence. Do not request or assume access to raw rows or heldout results."
    )


def call_api(key: str, model: str, payload: dict):
    body = {
        "model": model,
        "instructions": instructions(),
        "input": json.dumps(payload, ensure_ascii=False),
        "text": {"format": {"type": "json_schema", "name": "subgroup_graph_selection", "strict": True, "schema": OUTPUT_SCHEMA}},
        "max_output_tokens": max(3000, 240 * len(payload["candidate_edges"])),
        "store": False,
    }
    started = time.perf_counter()
    response = api_json("https://api.openai.com/v1/responses", key, body, timeout=300)
    return json.loads(output_text(response)), response.get("usage", {}), time.perf_counter() - started


def with_retries(fn):
    error = None
    for attempt in range(5):
        try:
            parsed, usage, latency = fn()
            return parsed, usage, latency, None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:800]
            error = f"HTTPError {exc.code}: {body}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(30.0, 2 ** attempt + random.random()))
    return {}, {}, math.nan, error


def metrics(frame: pd.DataFrame, action_column: str, name: str) -> dict:
    active = frame[action_column].isin(["activate_in_group", "deactivate_in_group_as_exception"])
    gold_safe = frame.replicated.astype(bool) & ~frame.sensitive_or_demographic.astype(bool)
    correct_direction = (
        (frame[action_column].eq("activate_in_group") & frame.graph_interpretation.eq("edge_active_in_subgroup"))
        | (frame[action_column].eq("deactivate_in_group_as_exception") & frame.graph_interpretation.eq("exception_subgroup"))
    )
    return {
        "method": name,
        "candidates": len(frame),
        "actuated": int(active.sum()),
        "actuation_precision_replicated_safe": float(gold_safe[active].mean()) if active.any() else None,
        "actuation_recall_replicated_safe": float(active[gold_safe].mean()) if gold_safe.any() else None,
        "direction_accuracy_on_true_actuation": float(correct_direction[active & gold_safe].mean()) if (active & gold_safe).any() else None,
        "unreplicated_actuation_rate": float(active[~frame.replicated.astype(bool)].mean()) if (~frame.replicated.astype(bool)).any() else 0.0,
        "sensitive_actuation_rate": float(active[frame.sensitive_or_demographic.astype(bool)].mean()) if frame.sensitive_or_demographic.astype(bool).any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")
    payloads = json.loads((args.audit / "llm_train_only_payload.json").read_text())
    heldout = pd.read_csv(args.audit / "all_subgroup_edge_candidates.csv")
    response_rows, raw = [], []
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_latency = 0.0
    for payload in payloads:
        if payload.get("privacy") != "schema_and_train_aggregates_only":
            raise RuntimeError("refusing API call: payload privacy boundary not verified")
        parsed, usage, latency, error = with_retries(lambda: call_api(key, args.model, payload))
        raw.append({"dataset": payload["dataset"], "response": parsed, "usage": usage, "latency": latency, "error": error})
        if error is None:
            total_latency += latency
            for key_name in totals:
                totals[key_name] += int(usage.get(key_name, 0) or 0)
        returned = {item.get("candidate_id"): item for item in parsed.get("decisions", [])}
        for candidate in payload["candidate_edges"]:
            decision = returned.get(candidate["candidate_id"], {})
            response_rows.append({
                "dataset": payload["dataset"], "candidate_id": candidate["candidate_id"],
                "action": decision.get("action", "missing"),
                "confidence": decision.get("confidence"),
                "semantic_rationale": decision.get("semantic_rationale", ""),
                "missing": not bool(decision), "error": error or "",
            })
    decisions = pd.DataFrame(response_rows)
    decisions.to_csv(args.output / "llm_decisions.csv", index=False)
    with (args.output / "api_responses.jsonl").open("w") as handle:
        for item in raw:
            handle.write(json.dumps(item) + "\n")
    evaluated = decisions.merge(heldout, on=["dataset", "candidate_id"], how="left", validate="one_to_one")
    if evaluated.replicated.isna().any():
        raise RuntimeError("candidate evaluation join failed")
    evaluated["statistics_action"] = evaluated.graph_interpretation.map({
        "edge_active_in_subgroup": "activate_in_group",
        "exception_subgroup": "deactivate_in_group_as_exception",
    })
    conservative = (
        ((evaluated.train_group_violation <= 0.05) & (evaluated.train_other_violation >= 0.15))
        | ((evaluated.train_other_violation <= 0.05) & (evaluated.train_group_violation >= 0.15))
    ) & ~evaluated.sensitive_or_demographic.astype(bool)
    evaluated["conservative_action"] = "soft_dependency_only"
    evaluated.loc[conservative, "conservative_action"] = evaluated.loc[conservative, "statistics_action"]
    evaluated.to_csv(args.output / "evaluated_decisions.csv", index=False)
    summaries = pd.DataFrame([
        metrics(evaluated, "statistics_action", "train_statistics_all"),
        metrics(evaluated, "conservative_action", "conservative_statistics"),
        metrics(evaluated, "action", f"llm_semantic_selector:{args.model}"),
    ])
    summaries.to_csv(args.output / "summary.csv", index=False)
    lines = ["# LLM subgroup-graph semantic selection", "", f"Model: `{args.model}`", "",
             "The API received schema and training aggregates only. Heldout replication labels were joined locally after all decisions.", "",
             "| Method | Candidates | Actuated | Precision | Recall | Direction accuracy | Unreplicated actuation | Sensitive actuation |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries.itertuples(index=False):
        def fmt(value): return "NA" if pd.isna(value) else f"{value:.4f}"
        lines.append(f"| {row.method} | {row.candidates} | {row.actuated} | {fmt(row.actuation_precision_replicated_safe)} | {fmt(row.actuation_recall_replicated_safe)} | {fmt(row.direction_accuracy_on_true_actuation)} | {fmt(row.unreplicated_actuation_rate)} | {fmt(row.sensitive_actuation_rate)} |")
    lines += ["", f"Token usage: `{totals}`; API latency: `{total_latency:.2f}s`; missing decisions: `{int(decisions.missing.sum())}`."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "status": "executed" if not decisions.missing.all() else "failed",
        "model": args.model, "privacy": "schema_and_train_aggregates_only",
        "heldout_sent_to_api": False, **totals, "latency_seconds": total_latency,
    }, indent=2))
    print(summaries.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
