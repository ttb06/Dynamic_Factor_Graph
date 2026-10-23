#!/usr/bin/env python3
"""Privacy-safe LCATS-style iterative graph discovery with calibrated edges.

Only schema, aggregate column summaries, and aggregate verifier feedback are
sent to the API. Candidate predicates are executed locally on train/heldout
splits; no real rows are transmitted or emitted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

from llm_feedback_audit import load_env
from semantic_llm_sweep import api_json, output_text


OPS = ["==", ">", ">=", "<", "<="]
PREDICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "column": {"type": "string"}, "operator": {"type": "string", "enum": OPS},
        "value": {"type": ["string", "number"]},
    },
    "required": ["column", "operator", "value"], "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "antecedent": PREDICATE_SCHEMA,
                    "consequent": PREDICATE_SCHEMA,
                    "semantic_kind": {"type": "string", "enum": ["domain", "logical", "temporal", "accounting", "process", "distributional"]},
                    "expected_hardness": {"type": "string", "enum": ["hard", "near_hard", "soft"]},
                    "external_rationale": {"type": "string"},
                },
                "required": ["description", "antecedent", "consequent", "semantic_kind", "expected_hardness", "external_rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["constraints"], "additionalProperties": False,
}


def aggregate_schema(train: pd.DataFrame, categorical: list[str], numerical: list[str]) -> dict:
    columns = []
    for column in train.columns:
        values = train[column]
        item = {"name": column, "missing_rate": round(float(values.isna().mean()), 6)}
        if column in numerical:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            item.update({"type": "numeric", "count": int(len(numeric))})
            if len(numeric):
                quantiles = numeric.quantile([0, .05, .25, .5, .75, .95, 1]).to_dict()
                item["quantiles"] = {str(key): round(float(value), 6) for key, value in quantiles.items()}
        else:
            counts = values.astype(str).value_counts()
            safe = counts[counts.ge(10)].head(15)
            item.update({"type": "categorical", "cardinality": int(values.nunique(dropna=False)),
                         "frequent_values": [{"value": str(key), "count": int(value)} for key, value in safe.items()]})
        columns.append(item)
    return {"privacy": "schema_and_aggregates_only", "rows": len(train), "columns": columns}


def evaluate_predicate(frame: pd.DataFrame, predicate: dict) -> np.ndarray:
    column, operator, literal = predicate["column"], predicate["operator"], predicate["value"]
    if column not in frame:
        raise KeyError(column)
    values = frame[column]
    if operator == "==":
        if pd.api.types.is_numeric_dtype(values):
            return np.isclose(pd.to_numeric(values, errors="coerce").to_numpy(float), float(literal), rtol=0, atol=1e-7)
        return values.astype(str).str.strip().eq(str(literal).strip()).to_numpy()
    numeric = pd.to_numeric(values, errors="coerce")
    threshold = float(literal)
    fn = {">": numeric.gt, ">=": numeric.ge, "<": numeric.lt, "<=": numeric.le}[operator]
    return fn(threshold).fillna(False).to_numpy()


def verify(frame: pd.DataFrame, candidate: dict) -> dict:
    antecedent = evaluate_predicate(frame, candidate["antecedent"])
    consequent = evaluate_predicate(frame, candidate["consequent"])
    support = int(antecedent.sum())
    violations = int((antecedent & ~consequent).sum())
    confidence = (support - violations) / support if support else math.nan
    base = float(consequent.mean())
    lift = confidence / base if base > 0 and np.isfinite(confidence) else math.nan
    return {"support": support, "violations": violations, "violation_rate": violations / support if support else math.nan,
            "confidence": confidence, "base_rate": base, "lift": lift}


def canonical(candidate: dict) -> str:
    return json.dumps({"a": candidate["antecedent"], "c": candidate["consequent"]}, sort_keys=True)


def call_api(key: str, model: str, payload: dict, maximum: int):
    instructions = (
        "You discover general executable inter-column constraints for synthetic tabular data. "
        "Use external domain knowledge plus the supplied aggregate evidence. Propose directional implications, "
        "not correlations restated in prose. The antecedent is the state predicate that activates an edge; the "
        "consequent is the expected relation endpoint. Include hard rules only when semantically defensible; use "
        "near_hard or soft for empirical dependencies. Do not use target leakage or sensitive demographic stereotypes. "
        "Do not ask for raw rows. Use exact schema column names and only the supported operators."
    )
    body = {
        "model": model, "instructions": instructions,
        "input": json.dumps(payload, ensure_ascii=False),
        "text": {"format": {"type": "json_schema", "name": "verified_graph_candidates", "strict": True, "schema": OUTPUT_SCHEMA}},
        "max_output_tokens": max(3000, maximum * 260), "store": False,
    }
    start = time.perf_counter()
    response = api_json("https://api.openai.com/v1/responses", key, body, timeout=300)
    return json.loads(output_text(response)), response.get("usage", {}), time.perf_counter() - start


def with_retries(fn):
    error = None
    for attempt in range(5):
        try:
            result, usage, latency = fn()
            return result, usage, latency, None
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:800]}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(30, 2 ** attempt + random.random()))
    return {}, {}, math.nan, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--per-round", type=int, default=12)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")
    sys.path.insert(0, str(args.repo / "analysis/llm_feedback_20260922/scripts"))
    import cross_dataset_constraint_audit as audit
    train, heldout, categorical, numerical, target = audit.load_dataset(args.repo, args.dataset)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train))
    cut = int((1 - args.calibration_fraction) * len(train))
    discovery, calibration = train.iloc[order[:cut]].copy(), train.iloc[order[cut:]].copy()
    schema = aggregate_schema(discovery, categorical, numerical)
    accepted, rejected, seen, raw = [], [], set(), []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for round_index in range(args.rounds):
        payload = {
            "privacy": "schema_and_aggregates_only", "dataset": args.dataset, "schema": schema,
            "target_column": target,
            "round": round_index + 1,
            "already_retained": [{"description": row["description"], "classification": row["classification"]} for row in accepted[-20:]],
            "rejected_or_soft_feedback": [{"description": row.get("description", ""), "reason": row.get("reason", "")} for row in rejected[-20:]],
            "request": f"Propose at most {args.per_round} new, nonduplicate executable directional constraints.",
        }
        result, usage, latency, error = with_retries(lambda: call_api(key, args.model, payload, args.per_round))
        raw.append({"round": round_index + 1, "response": result, "usage": usage, "latency": latency, "error": error})
        for name in usage_total:
            usage_total[name] += int(usage.get(name, 0) or 0)
        if error:
            break
        new_count = 0
        for candidate in result.get("constraints", []):
            key_name = canonical(candidate)
            if key_name in seen:
                continue
            seen.add(key_name)
            new_count += 1
            try:
                discovery_stats = verify(discovery, candidate)
                calibration_stats = verify(calibration, candidate)
                heldout_stats = verify(heldout, candidate)
            except Exception as exc:
                rejected.append({**candidate, "reason": f"execution_error:{type(exc).__name__}:{exc}"})
                continue
            columns = [candidate["antecedent"]["column"], candidate["consequent"]["column"]]
            row = {**candidate, "discovery": discovery_stats, "calibration": calibration_stats,
                   "heldout": heldout_stats}
            if columns[0] == columns[1]:
                row.update(classification="rejected_tautological", reason="same_column_implication")
                rejected.append(row)
            elif max(discovery_stats["base_rate"], calibration_stats["base_rate"]) >= .995:
                row.update(classification="rejected_vacuous", reason="consequent_almost_universal")
                rejected.append(row)
            elif target in columns:
                row.update(classification="audit_only_target", reason="target_column_excluded_from_guidance_graph")
                rejected.append(row)
            elif audit.is_sensitive(columns):
                row.update(classification="audit_only_sensitive", reason="sensitive_column_excluded_from_guidance_graph")
                rejected.append(row)
            elif min(discovery_stats["support"], calibration_stats["support"]) < args.min_support:
                row.update(classification="rejected_low_support", reason=(
                    f"discovery={discovery_stats['support']},calibration={calibration_stats['support']}"))
                rejected.append(row)
            elif discovery_stats["violation_rate"] == 0 and calibration_stats["violation_rate"] == 0:
                row.update(classification="strict_verified", reason="zero_discovery_and_calibration_violations")
                accepted.append(row)
            elif (max(discovery_stats["violation_rate"], calibration_stats["violation_rate"]) <= .05
                  and min(discovery_stats["lift"], calibration_stats["lift"]) >= 1.05):
                row.update(classification="near_hard_calibrated", reason="stable_on_discovery_and_calibration")
                accepted.append(row)
            elif (min(discovery_stats["confidence"], calibration_stats["confidence"]) >= .70
                  and min(discovery_stats["lift"], calibration_stats["lift"]) >= 1.20):
                row.update(classification="soft_dependency", reason="stable_distributional_edge_not_hard_validity")
                accepted.append(row)
            else:
                row.update(classification="rejected_weak", reason=(
                    f"discovery_violation={discovery_stats['violation_rate']:.3f},"
                    f"calibration_violation={calibration_stats['violation_rate']:.3f}"))
                rejected.append(row)
        if not new_count:
            break
    (args.output / "verified_graph.json").write_text(json.dumps(accepted, indent=2))
    (args.output / "rejected_graph.json").write_text(json.dumps(rejected, indent=2))
    with (args.output / "api_rounds.jsonl").open("w") as handle:
        for item in raw:
            handle.write(json.dumps(item) + "\n")
    flat = []
    for row in accepted:
        flat.append({
            "description": row["description"], "classification": row["classification"],
            "semantic_kind": row["semantic_kind"], "expected_hardness": row["expected_hardness"],
            "antecedent": json.dumps(row["antecedent"], sort_keys=True),
            "consequent": json.dumps(row["consequent"], sort_keys=True),
            **{f"discovery_{key}": value for key, value in row["discovery"].items()},
            **{f"calibration_{key}": value for key, value in row["calibration"].items()},
            **{f"heldout_{key}": value for key, value in row["heldout"].items()},
        })
    pd.DataFrame(flat).to_csv(args.output / "verified_graph.csv", index=False)
    counts = pd.Series([row["classification"] for row in accepted]).value_counts().to_dict()
    rejected_counts = pd.Series([row["classification"] for row in rejected if "classification" in row]).value_counts().to_dict()
    summary = {"dataset": args.dataset, "model": args.model, "retained": len(accepted), "rejected": len(rejected),
               "classification_counts": counts, "rejected_classification_counts": rejected_counts,
               "usage": usage_total, "privacy": "schema_and_aggregates_only",
               "discovery_rows": len(discovery), "calibration_rows": len(calibration),
               "heldout_rows": len(heldout), "heldout_used_for_selection": False,
               "target_and_sensitive_edges_actuated": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# LLM + dataset verified graph discovery", "", f"- Dataset: `{args.dataset}`", f"- Model: `{args.model}`",
             f"- Retained: **{len(accepted)}**", f"- Rejected: **{len(rejected)}**", f"- Tokens: **{usage_total['total_tokens']}**",
             "", "| Classification | Count |", "| --- | ---: |"]
    for name, count in sorted(counts.items()):
        lines.append(f"| {name} | {count} |")
    lines += ["", "No raw real rows were sent to the API. Candidate retention used disjoint discovery and calibration partitions; heldout statistics were withheld until final local evaluation. Target and sensitive edges were audit-only."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
