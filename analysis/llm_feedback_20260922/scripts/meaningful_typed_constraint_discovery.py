#!/usr/bin/env python3
"""Discover non-redundant typed constraints and assign conservative authority.

This script intentionally separates four questions that older graph discovery
jobs conflated:

1. Does a compact functional program fit the discovery split?
2. Does the program survive counterexample/calibration splits and a
   permutation-null test?
3. Does an LLM find a defensible measurement/process interpretation?
4. Is there independent documentation that permits generation-time actuation?

Statistical evidence can retrieve or score an edge, but can never make it a
hard actuator.  LLM review can prioritize source verification, but an LLM is
not itself an authoritative source.  Only a match to the separately supplied
documented catalog receives ``usage_role=actuate``.

No held-out row is used to propose or select a candidate.  No real row is sent
to an external API; reviewers receive schema names and aggregate diagnostics.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from knowledge_graph_protocol import retry_call
from unorouter_env import install_environment


STAT = {"min": "min", "minimum": "min", "avg": "avg", "average": "avg",
        "mean": "avg", "max": "max", "maximum": "max"}
SEMANTIC_DECISIONS = [
    "likely_documented_invariant", "semantic_soft_score", "statistical_only", "reject"
]


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def clean_name(column: str) -> str:
    return re.sub(r"sharess\b", "shares", column.strip().lower())


def tokens(column: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", clean_name(column))


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def robust_scale(frame: pd.DataFrame, columns: list[str]) -> float:
    scales = []
    for column in columns:
        values = numeric(frame, column)
        finite = values[np.isfinite(values)]
        if finite.size:
            q25, q75 = np.quantile(finite, [.25, .75])
            scales.append(max(float(q75-q25), float(np.median(np.abs(finite))), 1.0))
    return max(scales or [1.0])


def program_columns(program: dict) -> list[str]:
    kind = program["kind"]
    if kind in {"ordered_triplet", "sum_equal", "zero_cooccurrence"}:
        return list(program["columns"])
    if kind == "abs_equal":
        return [program["source"], program["target_column"]]
    if kind == "less_equal":
        return [program["left"], program["right"]]
    raise ValueError(kind)


def evaluate_program(frame: pd.DataFrame, program: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return applicable, satisfied, non-negative residual arrays."""
    kind = program["kind"]
    tolerance = float(program.get("tolerance", 1e-8))
    if kind == "ordered_triplet":
        low, avg, high = (numeric(frame, column) for column in program["columns"])
        applicable = np.isfinite(low) & np.isfinite(avg) & np.isfinite(high)
        residual = np.maximum(np.maximum(low-avg, avg-high), 0.0)
    elif kind == "sum_equal":
        matrix = np.column_stack([numeric(frame, column) for column in program["columns"]])
        applicable = np.isfinite(matrix).all(axis=1)
        gate = program.get("gate_sum_columns", [])
        if gate:
            gate_values = np.column_stack([numeric(frame, column) for column in gate])
            applicable &= np.isfinite(gate_values).all(axis=1)
            applicable &= gate_values.sum(axis=1) > float(program.get("gate_sum_gt", 1e-12))
        residual = np.abs(matrix.sum(axis=1) - float(program["target"]))
    elif kind == "abs_equal":
        source = numeric(frame, program["source"])
        target = numeric(frame, program["target_column"])
        applicable = np.isfinite(source) & np.isfinite(target)
        residual = np.abs(target - np.abs(source-float(program.get("center", 0.0))))
    elif kind == "less_equal":
        left, right = numeric(frame, program["left"]), numeric(frame, program["right"])
        applicable = np.isfinite(left) & np.isfinite(right)
        residual = np.maximum(left-right, 0.0)
    elif kind == "zero_cooccurrence":
        matrix = np.column_stack([numeric(frame, column) for column in program["columns"]])
        applicable = np.isfinite(matrix).all(axis=1)
        zero = np.isclose(matrix, 0.0, atol=tolerance)
        residual = (~(zero.all(axis=1) | (~zero).all(axis=1))).astype(float)
        tolerance = 0.0
    else:
        raise ValueError(f"unsupported program kind {kind}")
    residual = np.where(np.isfinite(residual), residual, np.inf)
    satisfied = residual <= tolerance
    return applicable, satisfied, residual


def metrics(frame: pd.DataFrame, program: dict) -> dict:
    applicable, satisfied, residual = evaluate_program(frame, program)
    support = int(applicable.sum())
    bad = int((applicable & ~satisfied).sum())
    active_residual = residual[applicable]
    return {
        "rows": len(frame), "support": support,
        "coverage": support/max(len(frame), 1), "violations": bad,
        "violation_rate": bad/support if support else math.nan,
        "median_residual": float(np.median(active_residual)) if support else math.nan,
        "p95_residual": float(np.quantile(active_residual, .95)) if support else math.nan,
    }


def permuted_violation(frame: pd.DataFrame, program: dict, seed: int, repeats: int = 5) -> float:
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(repeats):
        shuffled = frame.copy()
        columns = program_columns(program)
        # Preserve every marginal but destroy row-wise dependence.
        for column in columns[1:]:
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
        rate = metrics(shuffled, program)["violation_rate"]
        if np.isfinite(rate): rates.append(rate)
    return float(np.mean(rates)) if rates else math.nan


def bootstrap_stability(frame: pd.DataFrame, program: dict, seed: int, repeats: int = 20) -> dict:
    rng = np.random.default_rng(seed); rates = []
    for _ in range(repeats):
        sample = frame.iloc[rng.integers(0, len(frame), size=len(frame))]
        rate = metrics(sample, program)["violation_rate"]
        if np.isfinite(rate): rates.append(rate)
    return {
        "mean_violation_rate": float(np.mean(rates)) if rates else math.nan,
        "sd_violation_rate": float(np.std(rates)) if rates else math.nan,
        "p95_violation_rate": float(np.quantile(rates, .95)) if rates else math.nan,
    }


def candidate(candidate_id: str, description: str, program: dict, family: str,
              discovery_mechanism: str) -> dict:
    return {
        "candidate_id": candidate_id, "description": description,
        "columns": program_columns(program), "program": program,
        "relation_family": family, "discovery_mechanism": discovery_mechanism,
        "evidence_basis": "empirical_signal", "classification": "soft",
        "semantic_authority": False, "usage_role": "retrieve",
    }


def ordered_candidates(frame: pd.DataFrame, numerical: list[str]) -> list[dict]:
    groups: dict[tuple[str, ...], dict[str, str]] = defaultdict(dict)
    for column in numerical:
        parts = tokens(column)
        for index, part in enumerate(parts):
            if part not in STAT: continue
            signature = tuple(parts[:index] + ["{stat}"] + parts[index+1:])
            groups[signature][STAT[part]] = column
    result = []
    for ordinal, (signature, members) in enumerate(sorted(groups.items())):
        if not all(key in members for key in ["min", "avg", "max"]): continue
        columns = [members["min"], members["avg"], members["max"]]
        scale = robust_scale(frame, columns)
        program = {"kind": "ordered_triplet", "columns": columns,
                   "tolerance": max(1e-8, 1e-7*scale)}
        result.append(candidate(
            f"stat_order_{ordinal:03d}",
            f"Ordered summary: {columns[0].strip()} <= {columns[1].strip()} <= {columns[2].strip()}",
            program, "statistical_order_program", "column_name_template_and_split_test",
        ))
    return result


def abs_candidates(frame: pd.DataFrame, numerical: list[str]) -> list[dict]:
    by_clean = {clean_name(column): column for column in numerical}; result = []
    for ordinal, target in enumerate(sorted(numerical)):
        name = clean_name(target)
        if not name.startswith("abs_") or name[4:] not in by_clean: continue
        source = by_clean[name[4:]]
        x, y = numeric(frame, source), numeric(frame, target)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 100: continue
        choices = []
        for center in [0.0, .5, 1.0, -.5]:
            residual = np.abs(y[finite]-np.abs(x[finite]-center))
            choices.append((float(np.quantile(residual, .95)), center))
        _, center = min(choices)
        program = {"kind": "abs_equal", "source": source, "target_column": target,
                   "center": center, "tolerance": 1e-6}
        result.append(candidate(
            f"stat_abs_{ordinal:03d}",
            f"Derived absolute feature: {target.strip()} = abs({source.strip()} - {center:g})",
            program, "statistical_functional_program", "name_match_plus_center_search",
        ))
    return result


def sum_candidates(frame: pd.DataFrame, numerical: list[str]) -> list[dict]:
    groups: dict[str, list[str]] = defaultdict(list)
    for column in numerical:
        name = clean_name(column)
        indexed = re.match(r"^(.*?[_-])(\d+)$", name)
        if indexed: groups[f"indexed:{indexed.group(1)}"].append(column)
        if "_is_" in name: groups[f"onehot:{name.split('_is_', 1)[0]}"].append(column)
    result = []
    for ordinal, (name, columns) in enumerate(sorted(groups.items())):
        columns = sorted(set(columns), key=clean_name)
        if len(columns) < 3: continue
        matrix = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        finite = np.isfinite(matrix).all(axis=1)
        if finite.sum() < 100: continue
        sums = matrix[finite].sum(axis=1)
        nearest = min([0.0, 1.0, 100.0], key=lambda target: np.median(np.abs(sums-target)))
        program = {"kind": "sum_equal", "columns": columns, "target": nearest, "tolerance": 1e-6}
        result.append(candidate(
            f"stat_sum_group_{ordinal:03d}",
            f"Grouped components sum to {nearest:g}: {', '.join(x.strip() for x in columns)}",
            program, "statistical_sum_program", "indexed_or_indicator_name_group",
        ))

    # Discover conditional two-column simplexes only among bounded quantities
    # with at least two shared semantic name tokens.
    bounded = []
    for column in numerical:
        values = numeric(frame, column); finite = values[np.isfinite(values)]
        if finite.size >= 100 and np.quantile(finite, .001) >= -1e-7 and np.quantile(finite, .999) <= 1+1e-7:
            bounded.append(column)
    pair_ordinal = 0
    stop = {"positive", "negative", "min", "max", "avg", "mean", "global", "abs"}
    for left, right in itertools.combinations(bounded, 2):
        shared = (set(tokens(left)) & set(tokens(right))) - stop
        if len(shared) < 2: continue
        a, b = numeric(frame, left), numeric(frame, right)
        active = np.isfinite(a) & np.isfinite(b) & ((a+b) > 1e-12)
        if active.sum() < 100: continue
        if np.quantile(np.abs(a[active]+b[active]-1.0), .95) > 1e-5: continue
        program = {"kind": "sum_equal", "columns": [left, right], "target": 1.0,
                   "tolerance": 1e-6, "gate_sum_columns": [left, right], "gate_sum_gt": 1e-12}
        result.append(candidate(
            f"stat_sum_pair_{pair_ordinal:03d}",
            f"Conditional simplex: {left.strip()} + {right.strip()} = 1 when their sum is nonzero",
            program, "statistical_conditional_sum_program", "bounded_pair_name_and_residual_search",
        )); pair_ordinal += 1
    return result


def zero_component_candidates(frame: pd.DataFrame, numerical: list[str]) -> list[dict]:
    usable, masks = [], {}
    for column in numerical:
        values = numeric(frame, column); finite = np.isfinite(values)
        if finite.sum() < 100: continue
        mask = np.isclose(values, 0.0, atol=1e-12) & finite
        rate = float(mask[finite].mean())
        if .01 <= rate <= .95:
            usable.append(column); masks[column] = mask
    parent = {column: column for column in usable}
    def find(column: str) -> str:
        while parent[column] != column:
            parent[column] = parent[parent[column]]; column = parent[column]
        return column
    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b: parent[b] = a
    for left, right in itertools.combinations(usable, 2):
        shared = set(tokens(left)) & set(tokens(right))
        if not (shared - {"rate", "avg", "min", "max", "global", "words"}): continue
        union_mask = masks[left] | masks[right]
        jaccard = float((masks[left] & masks[right]).sum()/max(union_mask.sum(), 1))
        if jaccard >= .985: union(left, right)
    groups: dict[str, list[str]] = defaultdict(list)
    for column in usable: groups[find(column)].append(column)
    result = []
    for ordinal, columns in enumerate(sorted((sorted(value) for value in groups.values() if len(value) >= 2))):
        program = {"kind": "zero_cooccurrence", "columns": columns, "tolerance": 1e-12}
        result.append(candidate(
            f"stat_zero_component_{ordinal:03d}",
            "Shared zero/nonzero state: " + ", ".join(column.strip() for column in columns),
            program, "statistical_zero_state_component", "compressed_zero_mask_equivalence_component",
        ))
    return result


def inequality_candidates(frame: pd.DataFrame, numerical: list[str], excluded_pairs: set[frozenset]) -> list[dict]:
    result = []; ordinal = 0
    for left, right in itertools.permutations(numerical, 2):
        if frozenset([left, right]) in excluded_pairs: continue
        shared = set(tokens(left)) & set(tokens(right))
        if not shared: continue
        a, b = numeric(frame, left), numeric(frame, right)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() < 100: continue
        scale = robust_scale(frame, [left, right]); tolerance = max(1e-8, 1e-7*scale)
        satisfaction = (a[finite] <= b[finite]+tolerance)
        if satisfaction.mean() < .99: continue
        # Reject equality aliases and relations created solely by disjoint ranges.
        if np.isclose(a[finite], b[finite], atol=tolerance).mean() > .98: continue
        if np.quantile(a[finite], .75) <= np.quantile(b[finite], .25): continue
        program = {"kind": "less_equal", "left": left, "right": right, "tolerance": tolerance}
        result.append(candidate(
            f"stat_le_{ordinal:03d}", f"Directional bound: {left.strip()} <= {right.strip()}",
            program, "statistical_directional_bound", "semantic_name_overlap_and_split_test",
        )); ordinal += 1
    return result


def signature(program: dict) -> str:
    normalized = dict(program)
    if normalized["kind"] in {"sum_equal", "zero_cooccurrence"}:
        normalized["columns"] = sorted(normalized["columns"])
    normalized.pop("tolerance", None)
    return json.dumps(normalized, sort_keys=True)


def documented_match(candidate_: dict, documented: list[dict]) -> dict | None:
    candidate_program = candidate_["program"]
    for edge in documented:
        reference = edge.get("program")
        if not reference or reference.get("kind") != candidate_program.get("kind"): continue
        left, right = dict(candidate_program), dict(reference)
        left.pop("tolerance", None); right.pop("tolerance", None)
        # Gating metadata is allowed to be more explicit in the documented catalog.
        for key in ["gate_sum_columns", "gate_sum_gt"]:
            left.pop(key, None); right.pop(key, None)
        if left.get("kind") in {"sum_equal", "zero_cooccurrence"}:
            left["columns"] = sorted(left["columns"]); right["columns"] = sorted(right["columns"])
        if left == right:
            return {"candidate_id": edge.get("candidate_id"), "source": edge.get("source"),
                    "evidence_basis": edge.get("evidence_basis")}
    return None


REVIEW_SCHEMA = {
    "type": "object", "properties": {"reviews": {"type": "array", "items": {
        "type": "object", "properties": {
            "candidate_id": {"type": "string"},
            "decision": {"type": "string", "enum": SEMANTIC_DECISIONS},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "semantic_basis": {"type": "string", "enum": [
                "measurement_definition", "schema_name_only", "domain_process",
                "empirical_association", "ambiguous"]},
            "direction_valid": {"type": "boolean"},
            "legitimate_counterexample": {"type": "string"},
            "missing_authority": {"type": "string"},
            "rationale": {"type": "string"},
        }, "required": ["candidate_id", "decision", "confidence", "semantic_basis",
                         "direction_valid", "legitimate_counterexample", "missing_authority", "rationale"],
        "additionalProperties": False,
    }}}, "required": ["reviews"], "additionalProperties": False,
}

PROPOSAL_SCHEMA = {
    "type": "object", "properties": {"proposals": {"type": "array", "items": {
        "type": "object", "properties": {
            "description": {"type": "string"},
            "kind": {"type": "string", "enum": ["ordered_triplet", "sum_equal", "abs_equal", "less_equal"]},
            "columns": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 8},
            "source_column": {"type": "string"}, "target_column": {"type": "string"},
            "target_value": {"type": "number"}, "center": {"type": "number"},
            "gate_sum_columns": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "gate_sum_gt": {"type": "number"}, "tolerance": {"type": "number", "minimum": 0},
            "semantic_basis": {"type": "string", "enum": [
                "measurement_definition", "domain_process", "schema_name_only"]},
            "source_to_verify": {"type": "string"},
            "legitimate_counterexample": {"type": "string"},
            "why_not_correlation": {"type": "string"},
        }, "required": ["description", "kind", "columns", "source_column", "target_column",
                         "target_value", "center", "gate_sum_columns", "gate_sum_gt", "tolerance",
                         "semantic_basis", "source_to_verify", "legitimate_counterexample", "why_not_correlation"],
        "additionalProperties": False,
    }}}, "required": ["proposals"], "additionalProperties": False,
}


def llm_propose(frame: pd.DataFrame, numerical: list[str], existing: list[dict], key: str,
                model: str, maximum: int) -> tuple[list[dict], dict]:
    schema = []
    for column in numerical:
        values = numeric(frame, column); finite = values[np.isfinite(values)]
        schema.append({"column": column, "display_name": column.strip(), "type": "numeric",
                       "minimum": float(finite.min()) if finite.size else None,
                       "maximum": float(finite.max()) if finite.size else None,
                       "unique": int(pd.Series(finite).nunique())})
    instruction = (
        "Propose a small number of executable inter-column programs for the UCI Online News Popularity feature schema. "
        "Use external measurement meaning, not correlation: summary min/average/max order, component sums, deterministic "
        "derived features, or true subset bounds are appropriate. Do not propose target relations, arbitrary thresholds, "
        "marginal ranges, positivity, integer type, or duplicated programs listed as already_discovered. A proposal is only "
        "a hypothesis and must name the external documentation needed before actuation. Use exact original column strings. "
        "For ordered_triplet put exactly three ordered columns in columns. For sum_equal put all summed columns in columns. "
        "For abs_equal set source_column and target_column. For less_equal put [left,right] in columns. Fill irrelevant fields "
        "with empty strings, empty lists or zero. Return at most 12 diverse proposals. No raw rows are available."
    )
    payload = {"dataset": "news", "schema": schema,
               "already_discovered": [item["description"] for item in existing],
               "epistemic_rule": "LLM proposal never grants hard authority; every program is executed on held-out splits and checked against an independent source catalog."}
    result, usage, latency, error = retry_call(
        key, model, instruction, payload, PROPOSAL_SCHEMA,
        "typed_constraint_proposals", "medium", maximum=maximum,
    )
    call = {"stage": "typed_proposal", "model": model, "usage": usage,
            "latency": latency, "error": error, "result": result}
    if error: raise RuntimeError(error)
    known = set(numerical)
    aliases = {column.strip(): column for column in numerical}
    def resolve(column: str) -> str | None:
        return column if column in known else aliases.get(str(column).strip())
    compiled = []
    for ordinal, row in enumerate(result.get("proposals", [])):
        kind = row["kind"]
        columns = [resolve(column) for column in row["columns"]]
        if any(column is None for column in columns): continue
        try:
            if kind == "ordered_triplet":
                if len(columns) != 3: continue
                program = {"kind": kind, "columns": columns,
                           "tolerance": max(1e-8, 1e-7*robust_scale(frame, columns))}
                family = "llm_order_program"
            elif kind == "sum_equal":
                gate = [resolve(column) for column in row["gate_sum_columns"]]
                if any(column is None for column in gate) or not 2 <= len(columns) <= 8: continue
                program = {"kind": kind, "columns": columns, "target": row["target_value"],
                           "tolerance": 1e-6}
                if gate:
                    program.update(gate_sum_columns=gate, gate_sum_gt=row["gate_sum_gt"])
                family = "llm_sum_program"
            elif kind == "abs_equal":
                source, target = resolve(row["source_column"]), resolve(row["target_column"])
                if source is None or target is None: continue
                program = {"kind": kind, "source": source,
                           "target_column": target, "center": row["center"],
                           "tolerance": 1e-6}
                family = "llm_functional_program"
            elif kind == "less_equal":
                if len(columns) != 2: continue
                program = {"kind": kind, "left": columns[0], "right": columns[1],
                           "tolerance": max(1e-8, 1e-7*robust_scale(frame, columns))}
                family = "llm_directional_bound"
            else: continue
            compiled.append({**candidate(
                f"llm_typed_{ordinal:03d}", row["description"], program, family,
                "llm_external_semantic_proposal_then_deterministic_verification"),
                "llm_proposal_metadata": {key_: row[key_] for key_ in [
                    "semantic_basis", "source_to_verify", "legitimate_counterexample", "why_not_correlation"]},
            })
        except (KeyError, TypeError, ValueError):
            continue
    return compiled, call


def llm_review(candidates: list[dict], key: str, model: str, maximum: int) -> tuple[dict, list[dict]]:
    compact = [{
        "candidate_id": item["candidate_id"], "description": item["description"],
        "columns": [column.strip() for column in item["columns"]],
        "program": native(item["program"]), "relation_family": item["relation_family"],
        "discovery": item["discovery"], "counterexample": item["counterexample"],
        "calibration": item["calibration"], "permutation_violation_rate": item["permutation_violation_rate"],
    } for item in candidates]
    system = (
        "You are reviewing automatically discovered executable constraints for the UCI Online News Popularity schema. "
        "No raw rows are provided. Statistical stability and permutation contrast establish dependence, not semantic truth. "
        "Mark likely_documented_invariant only when column meanings strongly imply a measurement identity, subset bound, "
        "summary order, probability simplex, or deterministic derived feature; still name the external source needed before "
        "actuation. Mark semantic_soft_score for plausible process knowledge with legitimate exceptions, statistical_only "
        "for dependence without independent meaning, and reject for tautology, leakage, redundancy, wrong direction or likely "
        "artifact. Never authorize hard repair from empirical fit alone. Review every candidate_id exactly once."
    )
    calls, reviews = [], []
    personas = [
        "Act as a measurement-semantics and database-constraint expert.",
        "Act as a skeptical reviewer searching for preprocessing artifacts, accidental thresholds and legitimate counterexamples.",
    ]
    for index, persona in enumerate(personas):
        result, usage, latency, error = retry_call(
            key, model, system+" "+persona, {"dataset": "news", "candidates": compact},
            REVIEW_SCHEMA, f"typed_constraint_review_{index}", "medium", maximum=maximum,
        )
        calls.append({"stage": f"reviewer_{index}", "model": model, "usage": usage,
                      "latency": latency, "error": error, "result": result})
        if error: raise RuntimeError(error)
        for row in result.get("reviews", []):
            reviews.append({**row, "reviewer": index})

    # Consensus is intentionally deterministic; this avoids asking an LLM to
    # adjudicate its own authority.  Disagreement sends the edge to the most
    # conservative of the two roles.
    order = {"reject": 0, "statistical_only": 1, "semantic_soft_score": 2,
             "likely_documented_invariant": 3}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in reviews: grouped[row["candidate_id"]].append(row)
    consensus = {}
    for item in candidates:
        rows = grouped.get(item["candidate_id"], [])
        if len(rows) != 2:
            consensus[item["candidate_id"]] = {"decision": "reject", "confidence": 0.0,
                                                "reason": "incomplete_review"}
            continue
        decision = min((row["decision"] for row in rows), key=lambda value: order[value])
        consensus[item["candidate_id"]] = {
            "decision": decision, "confidence": float(min(row["confidence"] for row in rows)),
            "agreement": rows[0]["decision"] == rows[1]["decision"], "reviews": rows,
        }
    return consensus, calls


def write_report(output: Path, catalog: list[dict], proposed: int, documented_total: int,
                 old_audit: dict, api_calls: list[dict]) -> None:
    retained = [item for item in catalog if item["usage_role"] != "reject"]
    role_counts = Counter(item["usage_role"] for item in catalog)
    family_counts = Counter(item["relation_family"] for item in retained)
    matched = sum(bool(item.get("documented_match")) for item in catalog)
    lines = [
        "# Meaningful typed-constraint discovery audit — News", "",
        "## Kết luận", "",
        f"Pipeline đề xuất **{proposed}** program, giữ **{len(retained)}** sau kiểm tra discovery/counterexample/calibration và semantic review.",
        f"Chỉ **{role_counts.get('actuate', 0)}** cạnh có independent documented authority để actuation; các cạnh còn lại chỉ được dùng để score/retrieve/verify.",
        f"Recall đối với catalog documented độc lập: **{matched}/{documented_total}**. Held-out chỉ được mở sau khi catalog đã freeze.",
        "", "## Authority và redundancy", "",
        "- Statistical fit không còn được phép tự nâng thành hard constraint.",
        "- LLM chỉ phân loại ý nghĩa và chỉ ra nguồn còn thiếu; LLM không tự cấp authority để sửa dữ liệu.",
        "- Các implication zero→zero lặp lại được nén thành một zero-state component.",
        f"- Graph cũ: {json.dumps(old_audit, ensure_ascii=False)}", "",
        "## Catalog", "",
        "| id | family | role | max split violation | permutation gap | semantic decision | documented |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for item in catalog:
        max_violation = max(item[split]["violation_rate"] for split in ["discovery", "counterexample", "calibration"])
        lines.append(
            f"| {item['candidate_id']} | {item['relation_family']} | {item['usage_role']} | "
            f"{max_violation:.4f} | {item['permutation_gap']:.4f} | "
            f"{item.get('semantic_review', {}).get('decision', 'not_run')} | "
            f"{'yes' if item.get('documented_match') else 'no'} |"
        )
    lines.extend(["", "## Family coverage", "", "```json",
                  json.dumps(family_counts, indent=2), "```", "",
                  "## API", "", f"Calls: {len(api_calls)}. Raw rows sent: false.", ""])
    (output/"RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="news", choices=["news"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--llm-review", action="store_true")
    parser.add_argument("--llm-propose", action="store_true")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    parser.add_argument("--documented-graph", type=Path)
    parser.add_argument("--old-empirical-graph", type=Path)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--max-candidates", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.repo/"analysis/llm_feedback_20260922/scripts"))
    import cross_dataset_constraint_audit as audit
    train, heldout, categorical, numerical, target = audit.load_dataset(args.repo, args.dataset)
    numerical = [column for column in numerical if column != target]
    rng = np.random.default_rng(args.seed); order = rng.permutation(len(train)); n = len(train)
    discovery = train.iloc[order[:int(.60*n)]].copy()
    counterexample = train.iloc[order[int(.60*n):int(.80*n)]].copy()
    calibration = train.iloc[order[int(.80*n):]].copy()

    raw = []
    raw.extend(ordered_candidates(discovery, numerical))
    raw.extend(abs_candidates(discovery, numerical))
    raw.extend(sum_candidates(discovery, numerical))
    raw.extend(zero_component_candidates(discovery, numerical))
    ordered_pairs = {frozenset(pair) for item in raw if item["program"]["kind"] == "ordered_triplet"
                     for pair in itertools.combinations(item["columns"], 2)}
    raw.extend(inequality_candidates(discovery, numerical, ordered_pairs))

    calls = []; key = ""
    if args.llm_review or args.llm_propose:
        if not args.env_file: raise ValueError("--env-file required with an LLM stage")
        install_environment(args.env_file); key = os.environ.get("UNOROUTER_API_KEY", "")
    if args.llm_propose:
        proposed, call = llm_propose(
            discovery, numerical, raw, key, args.model, args.max_output_tokens)
        raw.extend(proposed); calls.append(call)

    deduplicated = {}
    for item in raw: deduplicated.setdefault(signature(item["program"]), item)
    audited, audit_rejections = [], []
    for index, item in enumerate(deduplicated.values()):
        item = {**item, "candidate_id": f"typed_{index:03d}"}
        item["discovery"] = metrics(discovery, item["program"])
        item["counterexample"] = metrics(counterexample, item["program"])
        item["calibration"] = metrics(calibration, item["program"])
        item["permutation_violation_rate"] = permuted_violation(
            discovery, item["program"], args.seed+index)
        item["permutation_gap"] = item["permutation_violation_rate"]-item["discovery"]["violation_rate"]
        item["bootstrap"] = bootstrap_stability(discovery, item["program"], args.seed+1000+index)
        rates = [item[split]["violation_rate"] for split in ["discovery", "counterexample", "calibration"]]
        support = min(item[split]["support"] for split in ["discovery", "counterexample", "calibration"])
        if support < 100 or any(not np.isfinite(rate) for rate in rates):
            audit_rejections.append({**item, "audit_rejection": "insufficient_or_nonfinite_split_support"})
            continue
        if max(rates) > .10:
            audit_rejections.append({**item, "audit_rejection": "split_violation_rate_above_10pct"})
            continue
        semantic_hypothesis = item["discovery_mechanism"] == \
            "llm_external_semantic_proposal_then_deterministic_verification"
        if item["permutation_gap"] < .05 and not semantic_hypothesis:
            audit_rejections.append({**item, "audit_rejection": "weak_permutation_contrast"})
            continue
        # A measurement-definition hypothesis may hold because marginal
        # supports are nested, so permutation need not destroy it.  Preserve
        # it for source review, but never let this exception grant authority.
        item["permutation_requirement_waived_for_source_review"] = bool(
            semantic_hypothesis and item["permutation_gap"] < .05)
        item["selection_score"] = (
            .50*(1-max(rates)) + .35*min(max(item["permutation_gap"], 0), 1)
            + .15*(1-min(item["bootstrap"]["sd_violation_rate"], 1))
        )
        audited.append(item)
    audited.sort(key=lambda item: (-item["selection_score"], item["candidate_id"]))
    frozen = audited[:args.max_candidates]

    consensus = {}
    if args.llm_review:
        consensus, review_calls = llm_review(frozen, key, args.model, args.max_output_tokens)
        calls.extend(review_calls)

    documented = json.loads(args.documented_graph.read_text()) if args.documented_graph else []
    for item in frozen:
        review = consensus.get(item["candidate_id"], {"decision": "not_run", "confidence": 0.0})
        item["semantic_review"] = review
        match = documented_match(item, documented)
        item["documented_match"] = match
        item["heldout"] = metrics(heldout, item["program"])
        if match:
            item.update(evidence_basis="documented_measurement_semantics_and_split_verification",
                        classification="hard" if item["heldout"]["violation_rate"] <= .01 else "near_hard",
                        semantic_authority=True, usage_role="actuate")
        elif review.get("decision") == "semantic_soft_score":
            item["usage_role"] = "score"
        elif review.get("decision") == "likely_documented_invariant":
            item["usage_role"] = "verify_source"
        elif review.get("decision") == "reject":
            item["usage_role"] = "reject"
        else:
            item["usage_role"] = "retrieve"

    old_audit = {}
    if args.old_empirical_graph and args.old_empirical_graph.exists():
        old = json.loads(args.old_empirical_graph.read_text())
        old_audit = {
            "edges": len(old),
            "empirical_edges_called_hard_or_near_hard": sum(
                edge.get("evidence_basis") == "empirical_signal"
                and edge.get("classification") in {"hard", "near_hard"} for edge in old),
            "zero_literal_edges": sum(" == 0" in edge.get("description", "") for edge in old),
        }

    metadata = {
        "dataset": args.dataset, "seed": args.seed,
        "split_rows": {"discovery": len(discovery), "counterexample": len(counterexample),
                       "calibration": len(calibration), "heldout": len(heldout)},
        "proposed": len(raw), "deduplicated": len(deduplicated), "selected": len(frozen),
        "llm_review": args.llm_review, "llm_propose": args.llm_propose,
        "model": args.model if (args.llm_review or args.llm_propose) else None,
        "api_calls": len(calls), "total_tokens": sum(
            int(call.get("usage", {}).get("total_tokens", 0) or 0) for call in calls),
        "real_rows_sent": False, "heldout_used_for_selection": False,
        "authority_policy": "statistics_retrieve_or_score; llm_prioritizes; independent_documentation_actuates",
    }
    (args.output/"catalog.json").write_text(json.dumps(native(frozen), indent=2))
    (args.output/"audit_rejections.json").write_text(json.dumps(native(audit_rejections), indent=2))
    (args.output/"actuation_graph.json").write_text(json.dumps(native([
        item for item in frozen if item["usage_role"] == "actuate"
    ]), indent=2))
    (args.output/"semantic_review_queue.json").write_text(json.dumps(native([
        item for item in frozen if item["usage_role"] != "actuate"
    ]), indent=2))
    (args.output/"api_calls.json").write_text(json.dumps(native(calls), indent=2))
    (args.output/"run_metadata.json").write_text(json.dumps(native(metadata), indent=2))
    write_report(args.output, frozen, len(raw), len(documented), old_audit, calls)
    print(json.dumps(native({**metadata, "roles": Counter(item["usage_role"] for item in frozen)}), indent=2))


if __name__ == "__main__":
    main()
