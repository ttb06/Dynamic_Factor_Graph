#!/usr/bin/env python3
"""Multi-stage, privacy-safe protocol for generating verified tabular knowledge graphs.

Protocol:
  schema/aggregate evidence -> independent proposal roles -> typed compilation
  -> deterministic checks -> counterexample-partition critique -> independent
  semantic judge -> blind calibration -> frozen heldout audit.

No real row is sent to an external API or written to the output artifacts.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from semantic_llm_sweep import api_json, output_text
from unorouter_env import install_environment


OPS = ["==", "!=", ">", ">=", "<", "<="]
FAMILIES = ["logical", "temporal", "accounting", "process", "missingness", "distributional", "domain", "representation"]
HARDNESS = ["hard", "near_hard", "soft"]
EVIDENCE = ["external_domain", "schema_semantics", "empirical_signal", "hybrid"]
ISSUES = [
    "none", "tautology", "vacuous", "spurious_correlation", "normative_or_sensitive",
    "wrong_direction", "unsupported_hardness", "low_support", "type_error",
    "missing_exception", "dataset_specific_codebook", "ambiguous_semantics",
]

PREDICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "left_column": {"type": "string"},
        "operator": {"type": "string", "enum": OPS},
        "right_kind": {"type": "string", "enum": ["literal", "column"]},
        "right_column": {"type": "string"},
        "right_value": {"type": ["string", "number", "boolean"]},
        "tolerance": {"type": "number", "minimum": 0},
    },
    "required": ["left_column", "operator", "right_kind", "right_column", "right_value", "tolerance"],
    "additionalProperties": False,
}
EXCEPTION_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        "all_of": {"type": "array", "items": PREDICATE_SCHEMA, "minItems": 1, "maxItems": 3},
        "reason": {"type": "string"},
    },
    "required": ["all_of", "reason"],
    "additionalProperties": False,
}
CANDIDATE_PROPERTIES = {
    "description": {"type": "string"},
    "applicability": {"type": "array", "items": PREDICATE_SCHEMA, "maxItems": 3},
    "requirements": {"type": "array", "items": PREDICATE_SCHEMA, "minItems": 1, "maxItems": 3},
    "exceptions": {"type": "array", "items": EXCEPTION_GROUP_SCHEMA, "maxItems": 3},
    "relation_family": {"type": "string", "enum": FAMILIES},
    "hardness_prior": {"type": "string", "enum": HARDNESS},
    "evidence_basis": {"type": "string", "enum": EVIDENCE},
    "rationale": {"type": "string"},
    "directionality_reason": {"type": "string"},
    "plausible_counterexample": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
}
CANDIDATE_REQUIRED = list(CANDIDATE_PROPERTIES)
PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object", "properties": CANDIDATE_PROPERTIES,
                "required": CANDIDATE_REQUIRED, "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"], "additionalProperties": False,
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["keep", "revise", "reject"]},
                    **CANDIDATE_PROPERTIES,
                    "semantic_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "issue_codes": {"type": "array", "items": {"type": "string", "enum": ISSUES}},
                    "critique": {"type": "string"},
                },
                "required": ["candidate_id", "decision", *CANDIDATE_REQUIRED,
                             "semantic_confidence", "issue_codes", "critique"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reviews"], "additionalProperties": False,
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["accept_hard", "accept_near_hard", "accept_soft", "reject"]},
                    "semantic_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "generalizability": {"type": "string", "enum": ["domain_general", "dataset_semantic", "dataset_empirical", "not_generalizable"]},
                    "issue_codes": {"type": "array", "items": {"type": "string", "enum": ISSUES}},
                    "rationale": {"type": "string"},
                },
                "required": ["candidate_id", "decision", "semantic_confidence", "generalizability", "issue_codes", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgments"], "additionalProperties": False,
}

DATASET_CONTEXT = {
    "adult": "Census-derived employment and income records. The income column is the prediction target and must not enter the guidance graph.",
    "default": "Credit-card account, repayment-status, bill and payment histories. PAY_0 is the current/most recent status and PAY_2..PAY_6 are historical months. The default label is the prediction target.",
    "shoppers": "Online shopping session behavior with page counts, durations, traffic context, bounce/exit rates, and purchase outcome. Revenue is the prediction target.",
    "news": "Online-news article metadata and content statistics. The shares column is the prediction target and must not enter the guidance graph. Count, minimum, average, and maximum fields describe related article or self-reference quantities, but only documented set/subset and summary-statistic semantics can authorize a hard relation.",
}

# Dataset documentation often names columns without specifying the causal or
# accounting linkage that an LLM is tempted to infer.  These caveats are part
# of the proposal input so absence of metadata cannot silently become a hard
# relation.  They are metadata only; no real rows are exposed.
CODEBOOK_CAVEATS = {
    "default": {
        "calendar": {
            "PAY_0": "repayment status in September 2005",
            "PAY_2": "repayment status in August 2005",
            "PAY_3": "repayment status in July 2005",
            "PAY_4": "repayment status in June 2005",
            "PAY_5": "repayment status in May 2005",
            "PAY_6": "repayment status in April 2005",
            "BILL_AMT1..BILL_AMT6": "statement amounts from September back to April 2005",
            "PAY_AMT1..PAY_AMT6": "amounts paid from September back to April 2005",
        },
        "unknown_or_ambiguous": [
            "The public codebook does not state which statement-cycle liability each PAY_AMTk settles.",
            "Values -2 and 0 occur in repayment-status fields but are not defined by the official UCI description; do not give them hard semantics without another authoritative source.",
            "The table omits refunds, write-offs, fees, payment channel, posting timestamp, and statement-cycle alignment.",
        ],
        "proposal_rule": "Relations depending on an unknown alignment or undocumented status code must be soft/diagnostic and name the ambiguity as a counterexample; they cannot be proposed as hard or near-hard actuation constraints.",
        "source": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
    },
    "news": {
        "known_measurement_semantics": [
            "n_tokens_title and n_tokens_content count title and article-content tokens.",
            "num_hrefs and num_self_hrefs count all links and self-referencing links.",
            "self_reference_min_shares, self_reference_avg_sharess, and self_reference_max_shares are summary statistics over self-referenced articles.",
            "LDA_00..LDA_04 are topic proportions; polarity and subjectivity columns are bounded scores or aggregate rates.",
        ],
        "unknown_or_ambiguous": [
            "A statistical association between popularity, topic, weekday, channel, or sentiment fields is not a semantic integrity constraint.",
            "Zero-valued summary fields may encode absence, preprocessing, or a true zero; do not invent missingness semantics.",
            "The public expert constraint list is withheld from proposal models and used only for an independent discovery-recall audit.",
        ],
        "proposal_rule": "Hard or near-hard authority requires a documented set/subset, min/mean/max, probability-simplex, or measurement identity. Other relations remain soft dependency guardrails even when empirically strong.",
        "source": "https://archive.ics.uci.edu/dataset/332/online+news+popularity",
    },
}

# Few-shot examples teach epistemic *roles*, not dataset-specific answers.
# They deliberately contrast executable semantics with plausible-looking
# correlations so that the model does not simply turn every train pattern into
# a hard repair rule.
SEMANTIC_PROGRAM_EXAMPLES = [
    {
        "pattern": "documented subset count",
        "observation": "self-reference links are a subset of all links",
        "program": "num_self_hrefs <= num_hrefs",
        "authority": "hard only when the codebook explicitly defines the subset",
        "trajectory_use": "actuator and satisfied-edge guardrail",
    },
    {
        "pattern": "directional process implication",
        "observation": "zero visited pages implies zero recorded duration, but zero duration need not imply zero pages",
        "program": "page_count == 0 -> page_duration == 0",
        "authority": "near-hard unless measurement semantics guarantee exactness",
        "trajectory_use": "conditional actuator; do not add the reverse edge",
    },
    {
        "pattern": "plausible but unsafe association",
        "observation": "a demographic label predicts an age range on train",
        "program": "reject as a validity constraint",
        "authority": "none: correlation, normative risk, and legitimate counterexamples",
        "trajectory_use": "never actuate",
    },
    {
        "pattern": "omitted process state",
        "observation": "a zero balance often accompanies zero payment, but refunds, posting delay, and cycle alignment are absent",
        "program": "conditional soft scoring edge with an explicit missing-information counterexample",
        "authority": "soft/diagnostic, never hard",
        "trajectory_use": "rank candidate actions; abstain from direct projection",
    },
]


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, usage: dict, latency: float):
        super().__init__(message)
        self.usage = usage
        self.latency = latency

COLUMN_GLOSSARY = {
    "adult": {
        "fnlwgt": "Census final sampling weight, not a person-level semantic measurement",
        "education.num": "numeric encoding of education level",
        "capital.gain": "reported capital gain", "capital.loss": "reported capital loss",
        "hours.per.week": "usual weekly working hours",
        "workclass": "employment sector", "occupation": "occupation category",
    },
    "default": {
        "LIMIT_BAL": "credit limit", "PAY_0": "most recent repayment status",
        "PAY_2..PAY_6": "repayment statuses in preceding months",
        "BILL_AMT1..BILL_AMT6": "monthly statement balances, newest to older",
        "PAY_AMT1..PAY_AMT6": "monthly payment amounts, newest to older",
    },
    "shoppers": {
        "Administrative/Informational/ProductRelated": "numbers of pages visited in each page family",
        "*_Duration": "time spent on the corresponding page family",
        "BounceRates": "session bounce-rate feature", "ExitRates": "session exit-rate feature",
        "PageValues": "page-value feature before completing a transaction",
        "SpecialDay": "proximity to a special day",
    },
    "news": {
        " n_tokens_title": "number of words in the title",
        " n_tokens_content": "number of words in the article content",
        " n_unique_tokens": "rate of unique words in the content",
        " n_non_stop_words": "rate of non-stop words in the content",
        " n_non_stop_unique_tokens": "rate of unique non-stop words in the content",
        " num_hrefs": "number of links",
        " num_self_hrefs": "number of links to other articles published by the same site",
        " self_reference_min_shares": "minimum shares among referenced same-site articles",
        " self_reference_avg_sharess": "average shares among referenced same-site articles",
        " self_reference_max_shares": "maximum shares among referenced same-site articles",
        " LDA_00..LDA_04": "topic proportions",
        " shares": "prediction target; excluded from graph discovery",
    },
}


def native(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, dict): return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [native(v) for v in value]
    return value


def literal_atom(column: str, operator: str, value: Any, tolerance: float = 0) -> dict:
    return {"left_column": column, "operator": operator, "right_kind": "literal",
            "right_column": "", "right_value": native(value), "tolerance": tolerance}


def column_atom(left: str, operator: str, right: str, tolerance: float = 0) -> dict:
    return {"left_column": left, "operator": operator, "right_kind": "column",
            "right_column": right, "right_value": 0, "tolerance": tolerance}


def predefined_candidates(dataset: str) -> list[dict]:
    """Small, auditable seed library; every seed still passes critic, calibration and heldout audit."""
    specifications = []
    if dataset == "adult":
        specifications = [
            ("Missing workclass co-occurs with missing occupation",
             [literal_atom("workclass", "==", "<MISSING_LIKE>")],
             [literal_atom("occupation", "==", "<MISSING_LIKE>")], "missingness", "near_hard"),
        ]
    elif dataset == "shoppers":
        for count, duration in [("Administrative", "Administrative_Duration"),
                                ("Informational", "Informational_Duration"),
                                ("ProductRelated", "ProductRelated_Duration")]:
            specifications.append((
                f"Zero {count} page count implies zero {duration}",
                [literal_atom(count, "==", 0)], [literal_atom(duration, "==", 0)],
                "process", "hard",
            ))
        specifications.append((
            "BounceRates should not exceed ExitRates", [],
            [column_atom("BounceRates", "<=", "ExitRates")], "domain", "near_hard",
        ))
    elif dataset == "default":
        specifications = [
            ("A positive most-recent delinquency status generally requires a positive current bill",
             [literal_atom("PAY_0", ">", 0)], [literal_atom("BILL_AMT1", ">", 0)],
             "accounting", "soft"),
            ("A non-positive preceding bill generally has no following payment",
             [literal_atom("BILL_AMT2", "<=", 0)], [literal_atom("PAY_AMT1", "==", 0)],
             "temporal", "soft"),
        ]
    result = []
    for description, applicability, requirements, family, hardness in specifications:
        result.append({
            "description": description, "applicability": applicability, "requirements": requirements,
            "exceptions": [], "relation_family": family, "hardness_prior": hardness,
            "evidence_basis": "schema_semantics", "rationale": "pre-registered interpretable hypothesis",
            "directionality_reason": "derived from the documented measurement/process direction",
            "plausible_counterexample": "measurement noise, delayed updates, or a legitimate exceptional process state",
            "confidence": .65, "proposal_role": "predefined_seed", "proposal_round": 0,
        })
    return result


def aggregate_schema(frame: pd.DataFrame, categorical: list[str], numerical: list[str]) -> dict:
    columns = []
    for column in frame.columns:
        values = frame[column]
        item = {"name": column, "missing_rate": round(float(values.isna().mean()), 6)}
        if column in numerical:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            item.update(type="numeric", count=int(len(numeric)), unique=int(numeric.nunique()))
            if len(numeric):
                item["quantiles"] = {str(q): round(float(v), 6) for q, v in numeric.quantile([0, .05, .25, .5, .75, .95, 1]).items()}
        else:
            counts = values.astype(str).str.strip().value_counts()
            item.update(type="categorical", cardinality=int(values.nunique(dropna=False)),
                        frequent_values=[{"value": str(k), "count": int(v)} for k, v in counts[counts.ge(8)].head(20).items()])
        columns.append(item)
    return {"privacy": "schema_and_aggregates_only", "rows": len(frame), "columns": columns}


def canonical_categorical(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    missing = result.isna() | result.str.lower().isin({"", "?", "na", "n/a", "nan", "unknown"})
    return result.mask(missing, "<MISSING_LIKE>")


def predicate_mask(frame: pd.DataFrame, predicate: dict) -> np.ndarray:
    left_column, operator = predicate["left_column"], predicate["operator"]
    if left_column not in frame:
        raise KeyError(left_column)
    left = frame[left_column]
    if predicate["right_kind"] == "column":
        right_column = predicate["right_column"]
        if right_column not in frame:
            raise KeyError(right_column)
        right = frame[right_column]
    else:
        right = predicate["right_value"]
    tolerance = float(predicate.get("tolerance", 0))
    if operator in {"==", "!="}:
        if pd.api.types.is_numeric_dtype(left):
            left_values = pd.to_numeric(left, errors="coerce").to_numpy(float)
            if isinstance(right, pd.Series):
                right_values = pd.to_numeric(right, errors="coerce").to_numpy(float)
            else:
                right_values = float(right)
            result = np.isclose(left_values, right_values, rtol=0, atol=max(tolerance, 1e-7), equal_nan=False)
        else:
            left_values = canonical_categorical(left)
            right_values = canonical_categorical(right) if isinstance(right, pd.Series) else str(right).strip()
            result = left_values.eq(right_values).fillna(False).to_numpy()
        return ~result if operator == "!=" else result
    left_values = pd.to_numeric(left, errors="coerce")
    right_values = pd.to_numeric(right, errors="coerce") if isinstance(right, pd.Series) else float(right)
    if operator == "<=": result = left_values.le(right_values + tolerance)
    elif operator == "<": result = left_values.lt(right_values + tolerance)
    elif operator == ">=": result = left_values.ge(right_values - tolerance)
    else: result = left_values.gt(right_values - tolerance)
    return result.fillna(False).to_numpy()


def verify(frame: pd.DataFrame, candidate: dict) -> dict[str, float]:
    active = np.ones(len(frame), dtype=bool)
    for item in candidate["applicability"]: active &= predicate_mask(frame, item)
    exception = np.zeros(len(frame), dtype=bool)
    for group in candidate.get("exceptions", []):
        group_mask = np.ones(len(frame), dtype=bool)
        for item in group["all_of"]: group_mask &= predicate_mask(frame, item)
        exception |= group_mask
    applicable = active & ~exception
    requirement = np.ones(len(frame), dtype=bool)
    for item in candidate["requirements"]: requirement &= predicate_mask(frame, item)
    support = int(applicable.sum()); violations = int((applicable & ~requirement).sum())
    confidence = (support - violations) / support if support else math.nan
    base = float(requirement.mean()); lift = confidence / base if base > 0 and np.isfinite(confidence) else math.nan
    return {"support": support, "violations": violations,
            "violation_rate": violations / support if support else math.nan,
            "confidence": confidence, "base_rate": base, "lift": lift,
            "coverage": support / len(frame), "exception_rate": float((active & exception).mean())}


def canonical(candidate: dict) -> str:
    predicates = sorted(candidate["applicability"], key=lambda x: json.dumps(x, sort_keys=True))
    requirements = sorted(candidate["requirements"], key=lambda x: json.dumps(x, sort_keys=True))
    exceptions = []
    for group in candidate.get("exceptions", []):
        exceptions.append({"all_of": sorted(group["all_of"], key=lambda x: json.dumps(x, sort_keys=True))})
    exceptions.sort(key=lambda x: json.dumps(x, sort_keys=True))
    return json.dumps({"applicability": predicates, "requirements": requirements, "exceptions": exceptions}, sort_keys=True)


def candidate_columns(candidate: dict) -> list[str]:
    exception_predicates = [p for group in candidate.get("exceptions", []) for p in group["all_of"]]
    atoms = candidate["applicability"] + candidate["requirements"] + exception_predicates
    columns = [p["left_column"] for p in atoms]
    columns.extend(p["right_column"] for p in atoms if p["right_kind"] == "column")
    return columns


def typecheck(candidate: dict, frame: pd.DataFrame, categorical: set[str], numerical: set[str],
              target: str | None, sensitive_fn) -> tuple[bool, str]:
    columns = candidate_columns(candidate)
    if any(column not in frame for column in columns): return False, "unknown_column"
    if target and target in columns: return False, "target_related"
    if sensitive_fn(columns): return False, "sensitive_related"
    if len(set(columns)) < 2:
        return False, "same_column_tautology"
    exception_predicates = [p for group in candidate.get("exceptions", []) for p in group["all_of"]]
    for predicate in candidate["applicability"] + exception_predicates + candidate["requirements"]:
        column, operator = predicate["left_column"], predicate["operator"]
        if predicate["right_kind"] == "column":
            right_column = predicate["right_column"]
            if not right_column or right_column not in frame: return False, "unknown_right_column"
            if right_column == column: return False, "self_comparison"
            if column in categorical or right_column in categorical:
                both_numeric = pd.api.types.is_numeric_dtype(frame[column]) and pd.api.types.is_numeric_dtype(frame[right_column])
                if not both_numeric and not (column in categorical and right_column in categorical and operator in {"==", "!="}):
                    return False, "incompatible_column_comparison"
            continue
        value = predicate["right_value"]
        if predicate["right_column"] not in {"", "<NONE>"}: return False, "literal_has_right_column"
        if column in categorical:
            if operator not in {"==", "!="} and not pd.api.types.is_numeric_dtype(frame[column]):
                return False, "categorical_operator"
            known = set(canonical_categorical(frame[column]).dropna().astype(str).unique())
            if operator in {"==", "!="} and str(value).strip() not in known: return False, "unknown_category"
            if operator not in {"==", "!="}:
                try: float(value)
                except Exception: return False, "numeric_literal"
        elif column in numerical:
            try: float(value)
            except Exception: return False, "numeric_literal"
        else: return False, "unknown_type"
    return True, "ok"


def api_call(key: str, model: str, instructions: str, payload: dict, schema: dict,
             name: str, effort: str, maximum: int = 30000) -> tuple[dict, dict, float]:
    # UnoRouter's public OpenAI-compatible surface currently supports strict
    # structured output reliably through Chat Completions.  The previous
    # Responses request was accepted by the gateway but returned an empty
    # payload for routed models, which made a valid graph-discovery run look
    # like a semantic failure.  Keep reasoning effort out of this portable
    # request: several non-OpenAI providers reject that provider-specific key.
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(native(payload), ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
        "max_tokens": maximum,
        "stream": False,
    }
    started = time.perf_counter()
    base_url = os.environ.get("UNOROUTER_BASE_URL", "https://api.unorouter.com/v1").rstrip("/")
    response = api_json(f"{base_url}/chat/completions", key, body, timeout=900)
    latency = time.perf_counter() - started
    if response.get("success") is False:
        raise StructuredOutputError(
            f"router error: {str(response.get('message', 'unknown error'))[:500]}", {}, latency
        )
    normalized = response.get("data") if isinstance(response.get("data"), dict) else response
    usage = normalized.get("usage", {})
    rendered = output_text(response)
    if not rendered:
        raise StructuredOutputError(
            f"empty structured output: status={normalized.get('status')} "
            f"incomplete_details={normalized.get('incomplete_details')} "
            f"top_level_keys={sorted(response)[:12]}", usage, latency
        )
    cleaned = rendered.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as first_exc:
        # Some routed providers prepend a short explanation despite requesting
        # JSON mode.  Accept only the single outer object; arbitrary prose is
        # never passed downstream as a constraint.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        try:
            if start < 0 or end <= start:
                raise first_exc
            parsed = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            snippet = cleaned[:240].replace("\n", "\\n")
            raise StructuredOutputError(
                f"invalid structured JSON: {exc}; status={response.get('status')}; prefix={snippet!r}", usage, latency
            ) from exc
    if isinstance(parsed, list):
        array_fields = [key for key, value in schema.get("properties", {}).items()
                        if value.get("type") == "array"]
        if len(array_fields) == 1:
            parsed = {array_fields[0]: parsed}
    if not isinstance(parsed, dict):
        raise StructuredOutputError(
            f"structured output has unsupported top-level type {type(parsed).__name__}", usage, latency
        )
    return parsed, usage, latency


def retry_call(*args, **kwargs) -> tuple[dict, dict, float, str | None]:
    error = None
    for attempt in range(5):
        try:
            result, usage, latency = api_call(*args, **kwargs)
            return result, usage, latency, None
        except StructuredOutputError as exc:
            # A routed provider occasionally returns a top-level error/empty
            # body without billing any tokens.  That is a transport failure,
            # not evidence that the model violated the requested schema.  It
            # is safe to retry because no candidate can enter the pipeline.
            # Conversely, malformed structured output with non-zero usage is
            # kept fail-closed: do not pay repeatedly and do not try to repair
            # an epistemically ambiguous constraint response in code.
            billed = int((exc.usage or {}).get("total_tokens", 0) or 0)
            message = str(exc)
            transient_empty = billed == 0 and (
                "empty structured output" in message
                or "top_level_keys=['error']" in message
            )
            if not transient_empty:
                return {}, exc.usage, exc.latency, message
            error = message
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:1000]}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}: break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(45, 2 ** attempt + random.random()))
    return {}, {}, math.nan, error


def empirical_hints(audit, dataset: str, discovery: pd.DataFrame, counterexample: pd.DataFrame,
                    categorical: list[str], numerical: list[str], target: str | None,
                    sensitive_fn, maximum: int = 30) -> tuple[list[dict], list[dict]]:
    """Mine only discovery data; counterexample statistics are not exposed to proposal models."""
    predictor_categorical = [c for c in categorical if c != target]
    predictor_numerical = [c for c in numerical if c != target]
    predicates = audit.predicate_bank(
        discovery, counterexample, predictor_categorical, predictor_numerical, min_support=30
    )
    mined = audit.mine_implications(
        dataset, discovery, counterexample, predicates, target,
        min_confidence=.70, max_per_antecedent=4,
    )
    hints, candidates = [], []
    predicate_by_id = {p.predicate_id: p for p in predicates}
    for row in mined:
        try:
            relation_id = row["constraint_id"].split(":", 1)[1]
            left_id, right_id = relation_id.split("->", 1)
            left, right = predicate_by_id[left_id], predicate_by_id[right_id]
            nodes = [
                literal_atom(left.column, "==" if left.operator == "eq" else left.operator, left.value),
                literal_atom(right.column, "==" if right.operator == "eq" else right.operator, right.value),
            ]
            candidate = {"description": str(row["constraint"]), "applicability": [nodes[0]], "requirements": [nodes[1]],
                         "exceptions": [], "relation_family": "distributional", "hardness_prior": "soft",
                         "evidence_basis": "empirical_signal", "rationale": "train-only candidate mining",
                         "directionality_reason": "empirical directional implication", "plausible_counterexample": "unknown",
                         "confidence": 0.5, "source": "empirical_mining"}
            columns = candidate_columns(candidate)
            if target and target in columns or sensitive_fn(columns): continue
            stats = verify(discovery, candidate)
            if stats["support"] < 30 or not np.isfinite(stats["lift"]) or stats["base_rate"] >= .995: continue
            score = math.log1p(stats["support"]) * min(stats["lift"], 8) * max(.01, 1-stats["violation_rate"])
            candidates.append({**candidate, "discovery": stats, "hint_score": score})
        except Exception: continue
    candidates.sort(key=lambda x: -x["hint_score"])
    seen_pairs = Counter()
    selected = []
    for item in candidates:
        pair = (item["applicability"][0]["left_column"], item["requirements"][0]["left_column"])
        if seen_pairs[pair] >= 2: continue
        seen_pairs[pair] += 1; selected.append(item)
        if len(selected) >= maximum: break
    for item in selected:
        hints.append({"relation": item["description"], "support": item["discovery"]["support"],
                      "confidence": round(item["discovery"]["confidence"], 5),
                      "base_rate": round(item["discovery"]["base_rate"], 5),
                      "lift": round(item["discovery"]["lift"], 5),
                      "violation_rate": round(item["discovery"]["violation_rate"], 5)})
    return hints, selected


def _fd_bin_count(values: pd.Series, fallback: int = 8, maximum: int = 16) -> int:
    """SAGE-compatible Freedman--Diaconis bin count with a small safe cap."""
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if finite.size < 2 or float(np.ptp(finite)) <= 0:
        return 2
    q25, q75 = np.percentile(finite, [25, 75]); iqr = float(q75 - q25)
    if iqr <= 0:
        return max(2, min(maximum, fallback))
    width = 2.0 * iqr * finite.size ** (-1.0 / 3.0)
    if width <= 1e-12:
        return max(2, min(maximum, fallback))
    return max(2, min(maximum, int(np.ceil(float(np.ptp(finite)) / width))))


def value_aware_dependency_hints(
    discovery: pd.DataFrame,
    categorical: list[str],
    numerical: list[str],
    target: str | None,
    sensitive_fn,
    maximum: int = 40,
) -> list[dict]:
    """Extract SAGE-style feature/value MI signals without calling an LLM.

    These are dependency-retrieval hints, never executable constraints.  The
    distinction is deliberate: high MI can identify where semantics may live,
    but it cannot establish direction, validity, or actuation authority.
    """
    columns = [c for c in discovery.columns if c != target and not sensitive_fn([c])]
    if not columns:
        return []
    discrete = pd.DataFrame(index=discovery.index)
    labels: dict[str, dict[int, str]] = {}
    for column in columns:
        values = discovery[column]
        if column in numerical and pd.to_numeric(values, errors="coerce").nunique(dropna=True) > 2:
            numeric = pd.to_numeric(values, errors="coerce")
            fill = float(numeric.median()) if numeric.notna().any() else 0.0
            bins = _fd_bin_count(numeric)
            estimator = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="uniform", subsample=None)
            transformed = estimator.fit_transform(numeric.fillna(fill).to_numpy().reshape(-1, 1)).astype(int).ravel()
            discrete[column] = transformed
            edges = estimator.bin_edges_[0]
            labels[column] = {
                index: f"[{edges[index]:.6g}, {edges[index + 1]:.6g}{']' if index == len(edges)-2 else ')'}"
                for index in range(len(edges) - 1)
            }
        else:
            encoded, uniques = pd.factorize(canonical_categorical(values), sort=True)
            discrete[column] = encoded
            labels[column] = {index: str(value) for index, value in enumerate(uniques)}

    signals: list[dict] = []
    def binary_discrete_mi(binary: np.ndarray, outcome: np.ndarray) -> float:
        """Exact empirical I(binary; outcome), avoiding thousands of sklearn fits."""
        outcome = np.asarray(outcome, dtype=int)
        if outcome.size == 0:
            return 0.0
        shifted = outcome - int(outcome.min())
        categories = int(shifted.max()) + 1
        joint = np.bincount(binary.astype(int) * categories + shifted,
                            minlength=2 * categories).reshape(2, categories).astype(float)
        joint /= joint.sum()
        px = joint.sum(axis=1, keepdims=True); py = joint.sum(axis=0, keepdims=True)
        expected = px @ py; mask = joint > 0
        return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))

    for conditioning in columns:
        condition_values = discrete[conditioning].to_numpy(int)
        for value in np.unique(condition_values):
            binary = (condition_values == value).astype(int)
            support = int(binary.sum())
            if support < 30 or support > len(binary) - 30:
                continue
            for current in columns:
                if current == conditioning:
                    continue
                score = binary_discrete_mi(binary, discrete[current].to_numpy(int))
                if score <= 0:
                    continue
                signals.append({
                    "current_feature": current,
                    "conditioning_feature": conditioning,
                    "conditioning_value_or_bin": labels[conditioning].get(int(value), str(value)),
                    "support": support,
                    "coverage": support / len(binary),
                    "mutual_information": score,
                    "authority": "retrieval_only_not_a_constraint",
                    "source": "SAGE_style_value_aware_MI",
                })
    signals.sort(key=lambda row: (-row["mutual_information"], -row["support"],
                                  row["current_feature"], row["conditioning_feature"]))
    selected: list[dict] = []
    per_pair: Counter = Counter()
    for signal in signals:
        pair = (signal["current_feature"], signal["conditioning_feature"])
        if per_pair[pair] >= 2:
            continue
        per_pair[pair] += 1; selected.append(signal)
        if len(selected) >= maximum:
            break
    return selected


def counterexample_signature(frame: pd.DataFrame, candidate: dict, target: str | None,
                             sensitive_fn, maximum: int = 4) -> list[dict]:
    """Privacy-safe aggregate contrasts for counterexample-guided revision."""
    active = np.ones(len(frame), dtype=bool)
    for predicate in candidate["applicability"]:
        active &= predicate_mask(frame, predicate)
    for group in candidate.get("exceptions", []):
        group_mask = np.ones(len(frame), dtype=bool)
        for predicate in group["all_of"]:
            group_mask &= predicate_mask(frame, predicate)
        active &= ~group_mask
    satisfied = np.ones(len(frame), dtype=bool)
    for predicate in candidate["requirements"]:
        satisfied &= predicate_mask(frame, predicate)
    bad, good = active & ~satisfied, active & satisfied
    if int(bad.sum()) < 10 or int(good.sum()) < 10:
        return []
    used = set(candidate_columns(candidate)); rows: list[dict] = []
    for column in frame.columns:
        if column == target or column in used or sensitive_fn([column]):
            continue
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce")
            bad_values, good_values = numeric[bad].dropna(), numeric[good].dropna()
            if len(bad_values) < 10 or len(good_values) < 10:
                continue
            scale = float(numeric.quantile(.75) - numeric.quantile(.25))
            if scale <= 1e-12:
                continue
            delta = float((bad_values.median() - good_values.median()) / scale)
            rows.append({"column": column, "kind": "numeric_median_contrast",
                         "standardized_bad_minus_good": delta,
                         "bad_median": float(bad_values.median()),
                         "satisfied_median": float(good_values.median()),
                         "score": abs(delta)})
        else:
            bad_top = canonical_categorical(values[bad]).value_counts(normalize=True).head(5)
            good_rates = canonical_categorical(values[good]).value_counts(normalize=True)
            for value, bad_rate in bad_top.items():
                good_rate = float(good_rates.get(value, 0.0))
                delta = float(bad_rate - good_rate)
                rows.append({"column": column, "kind": "categorical_enrichment",
                             "value": str(value), "bad_rate": float(bad_rate),
                             "satisfied_rate": good_rate, "score": abs(delta)})
    rows.sort(key=lambda row: -row["score"])
    return [{k: v for k, v in row.items() if k != "score"} for row in rows[:maximum]]


def classify_evidence(stats: list[dict], minimum: int, conditional: bool) -> str | None:
    if min(s["support"] for s in stats) < minimum: return None
    violation = max(s["violation_rate"] for s in stats)
    confidence = min(s["confidence"] for s in stats); lift = min(s["lift"] for s in stats)
    if not conditional:
        if violation <= .01: return "hard"
        if violation <= .10 and confidence >= .90: return "near_hard"
        if violation <= .30 and confidence >= .70: return "soft"
        return None
    if max(s["base_rate"] for s in stats) >= .995: return None
    if violation <= .01 and lift >= 1.01: return "hard"
    if violation <= .10 and confidence >= .90 and lift >= 1.02: return "near_hard"
    if violation <= .30 and confidence >= .70 and lift >= 1.20: return "soft"
    return None


def prepare_candidates(raw: list[dict], source: str, discovery: pd.DataFrame,
                       counterexample: pd.DataFrame, categorical: set[str], numerical: set[str],
                       target: str | None, sensitive_fn, prefix: str) -> tuple[list[dict], list[dict]]:
    valid, rejected, seen = [], [], set()
    for ordinal, item in enumerate(raw):
        if not isinstance(item, dict):
            rejected.append({"raw_type": type(item).__name__, "rejection": "malformed_schema"})
            continue
        missing = [key for key in CANDIDATE_REQUIRED if key not in item]
        if missing:
            rejected.append({"raw_candidate": native(item), "missing_fields": missing,
                             "rejection": "malformed_schema"})
            continue
        candidate = {key: item[key] for key in CANDIDATE_REQUIRED}
        key = canonical(candidate)
        if key in seen: rejected.append({**candidate, "rejection": "duplicate_within_source"}); continue
        seen.add(key)
        ok, reason = typecheck(candidate, discovery, categorical, numerical, target, sensitive_fn)
        if not ok: rejected.append({**candidate, "rejection": reason}); continue
        try:
            d, x = verify(discovery, candidate), verify(counterexample, candidate)
        except Exception as exc:
            rejected.append({**candidate, "rejection": f"execution:{exc}"}); continue
        if d["support"] < 10 or (candidate["applicability"] and d["base_rate"] >= .999):
            rejected.append({**candidate, "rejection": "discovery_low_support_or_vacuous", "discovery": d}); continue
        valid.append({**candidate, "candidate_id": f"{prefix}_{ordinal:03d}", "source": source,
                      "proposal_role": item.get("proposal_role"), "proposal_round": item.get("proposal_round"),
                      "canonical": key, "discovery": d, "counterexample": x})
    return valid, rejected


def select_after_calibration(candidates: list[dict], calibration: pd.DataFrame, heldout: pd.DataFrame,
                             minimum: int, require_judge: bool) -> tuple[list[dict], list[dict]]:
    frozen, rejected = [], []
    for item in candidates:
        if require_judge and item.get("judge_decision") == "reject":
            rejected.append({**item, "selection_rejection": "semantic_judge_reject"}); continue
        calibration_stats = verify(calibration, item)
        evidence_class = classify_evidence(
            [item["discovery"], item["counterexample"], calibration_stats], minimum,
            conditional=bool(item["applicability"]),
        )
        if evidence_class is None:
            rejected.append({**item, "calibration": calibration_stats, "selection_rejection": "blind_calibration_failure"}); continue
        if require_judge:
            judge_class = item["judge_decision"].replace("accept_", "")
            order = {"soft": 0, "near_hard": 1, "hard": 2}
            final_class = min([evidence_class, judge_class], key=lambda x: order[x])
            if item.get("judge_confidence", 0) < .60:
                rejected.append({**item, "calibration": calibration_stats, "selection_rejection": "low_semantic_confidence"}); continue
        else: final_class = evidence_class

        # Empirical fit is deliberately not an authority escalation path.  A
        # zero-violation association can still be a sampling accident,
        # preprocessing artefact, or redundant encoding.  Only externally or
        # schema-grounded semantics that passed the independent judge may be
        # used as an actuator.  Hybrid evidence is capped unless the judge
        # explicitly found a semantic (rather than empirical) basis.
        evidence_basis = item.get("evidence_basis", "empirical_signal")
        generalizability = item.get("generalizability", "dataset_empirical")
        semantic_authority = (
            require_judge
            and evidence_basis in {"external_domain", "schema_semantics", "hybrid"}
            and generalizability in {"domain_general", "dataset_semantic"}
        )
        if not semantic_authority:
            final_class = "soft"
        usage_role = "actuate" if semantic_authority and final_class in {"hard", "near_hard"} else "score"
        if evidence_basis == "empirical_signal":
            usage_role = "retrieve"
        heldout_stats = verify(heldout, item)
        frozen.append({**item, "calibration": calibration_stats, "classification": final_class,
                       "semantic_authority": semantic_authority, "usage_role": usage_role,
                       "heldout": heldout_stats,
                       "heldout_shift": abs(heldout_stats["violation_rate"]-calibration_stats["violation_rate"])
                       if heldout_stats["support"] else math.nan})
    return frozen, rejected


def graph_summary(name: str, items: list[dict], proposals: int) -> dict:
    shifts = [x["heldout_shift"] for x in items if np.isfinite(x.get("heldout_shift", math.nan))]
    families = Counter(x["relation_family"] for x in items)
    endpoints = set()
    for item in items:
        applicability_columns = set(candidate_columns({**item, "requirements": [], "exceptions": []}))
        requirement_columns = set(candidate_columns({**item, "applicability": [], "exceptions": []}))
        endpoints.add((tuple(sorted(applicability_columns)), tuple(sorted(requirement_columns))))
    return {"graph": name, "proposed": proposals, "retained": len(items),
            "hard": sum(x["classification"] == "hard" for x in items),
            "near_hard": sum(x["classification"] == "near_hard" for x in items),
            "soft": sum(x["classification"] == "soft" for x in items),
            "family_diversity": len(families), "endpoint_patterns": len(endpoints),
            "external_or_hybrid": sum(x["evidence_basis"] in {"external_domain", "hybrid"} for x in items),
            "heldout_stable_005": float(np.mean(np.asarray(shifts) <= .05)) if shifts else math.nan,
            "mean_heldout_shift": float(np.mean(shifts)) if shifts else math.nan}


def existing_graph_context(path: Path | None) -> dict | None:
    """Compact topology context for connectivity-aware proposal roles.

    This exposes typed programs and component membership, not real rows.  It is
    intentionally descriptive: an LLM may propose a bridge, but the ordinary
    discovery/counterexample/calibration and semantic-review stages still
    decide whether that bridge survives.
    """
    if path is None:
        return None
    edges = json.loads(path.read_text())
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    endpoint_patterns, edge_summaries = Counter(), []
    for edge in edges:
        columns = sorted(set(candidate_columns(edge)))
        for column in columns:
            find(column)
        for column in columns[1:]:
            union(columns[0], column)
        endpoint_patterns[tuple(columns)] += 1
        edge_summaries.append({
            "candidate_id": edge.get("candidate_id"),
            "columns": columns,
            "classification": edge.get("classification"),
            "usage_role": edge.get("usage_role", "unspecified"),
            "description": edge.get("description"),
        })
    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    components = sorted((sorted(value) for value in groups.values()), key=lambda value: (-len(value), value))
    return {
        "privacy": "typed_programs_and_aggregate_topology_only",
        "edge_count": len(edges),
        "components": components,
        "endpoint_patterns": [
            {"columns": list(columns), "edges": count}
            for columns, count in endpoint_patterns.most_common()
        ],
        "existing_edges": edge_summaries,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--catalog", type=Path, help="Deprecated: empirical relations are re-mined from discovery only.")
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--proposal-model", default="gpt-6.1-sol"); p.add_argument("--external-model", default="gpt-6-astra")
    p.add_argument("--critic-model", default="gpt-6.1-sol"); p.add_argument("--judge-model", default="gpt-6-astra")
    p.add_argument("--reasoning-effort", default="high"); p.add_argument("--per-role", type=int, default=14)
    p.add_argument("--judge-reasoning-effort", default="high")
    p.add_argument("--max-output-tokens", type=int, default=6000)
    p.add_argument("--total-token-budget", type=int, default=180000)
    p.add_argument("--critic-batch-size", type=int, default=8)
    p.add_argument("--judge-batch-size", type=int, default=12)
    p.add_argument("--max-review-candidates", type=int, default=0,
                   help="Deterministically cap protocol candidates before critic/judge; 0 keeps all.")
    p.add_argument("--max-judge-candidates", type=int, default=0,
                   help="Role-balanced cap after critique and before final semantic judgment; 0 keeps all.")
    p.add_argument("--proposal-profile", choices=["general", "complex_hyperedge", "trajectory_actionable", "connectivity_aware", "semantic_programmatic"], default="general")
    p.add_argument("--existing-graph", type=Path,
                   help="Optional typed graph whose topology is shown to connectivity-aware proposal roles.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--invalidate-stage-prefix", action="append", default=[])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--min-support", type=int, default=30); p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=args.resume)
    install_environment(args.env_file); key = os.environ.get("UNOROUTER_API_KEY")
    if not key: raise RuntimeError("UNOROUTER_API_KEY unavailable")
    sys.path.insert(0, str(args.repo/"analysis/llm_feedback_20260922/scripts"))
    import cross_dataset_constraint_audit as audit
    train, heldout, categorical_columns, numerical_columns, target = audit.load_dataset(args.repo, args.dataset)
    rng = np.random.default_rng(args.seed); order = rng.permutation(len(train)); n = len(train)
    discovery = train.iloc[order[:int(.60*n)]].copy()
    counterexample = train.iloc[order[int(.60*n):int(.80*n)]].copy()
    calibration = train.iloc[order[int(.80*n):]].copy()
    categorical, numerical = set(categorical_columns), set(numerical_columns)
    schema = aggregate_schema(discovery, categorical_columns, numerical_columns)
    hints, empirical_candidates = empirical_hints(
        audit, args.dataset, discovery, counterexample, categorical_columns, numerical_columns,
        target, audit.is_sensitive,
    )
    value_aware_hints = value_aware_dependency_hints(
        discovery, categorical_columns, numerical_columns, target, audit.is_sensitive,
    )
    seed_candidates = predefined_candidates(args.dataset)
    common = {"privacy": "schema_and_aggregates_only", "dataset": args.dataset,
              "dataset_context": DATASET_CONTEXT.get(args.dataset, args.dataset), "target_column": target,
              "column_glossary": COLUMN_GLOSSARY.get(args.dataset, {}),
              "codebook_caveats": CODEBOOK_CAVEATS.get(args.dataset, {}),
              "schema": schema, "allowed_operators": OPS,
              "value_aware_dependency_signals": value_aware_hints,
              "value_aware_signal_warning": (
                  "These SAGE-style train-only MI signals retrieve candidate feature/value contexts. "
                  "They are not validity constraints and cannot justify hardness or actuation."
              ),
              "predefined_hypotheses": [x["description"] for x in seed_candidates],
              "few_shot_epistemic_examples": (
                  SEMANTIC_PROGRAM_EXAMPLES if args.proposal_profile == "semantic_programmatic" else []
              ),
              "semantics": {"applicability": "all predicates are AND",
                            "requirements": "all requirement atoms are AND; right_kind=column enables column-to-column relations",
                            "atom_encoding": "literal atoms use right_column='' and right_value; column atoms use right_column and ignore right_value",
                            "exceptions": "predicates within all_of are AND; any matching exception group deactivates the edge"},
              "proposal_profile": args.proposal_profile,
              "existing_graph_context": existing_graph_context(args.existing_graph)}
    profile_instruction = ""
    if args.proposal_profile == "complex_hyperedge":
        profile_instruction = (
            " Prioritize executable conditional hyperedges rather than unary bounds or isolated pairwise correlations. "
            "A useful complex candidate has at least one of: multiple jointly necessary applicability atoms, multiple jointly "
            "necessary requirements, an explicit evidence-backed exception/deactivation group, or a column-to-column relation "
            "whose applicability depends on additional columns. Prefer three-or-more-column process, accounting, temporal, "
            "co-occurrence, and denial-constraint structures. Explain why each atom is necessary and whether a violation admits "
            "multiple plausible repair endpoints. Do not concatenate unrelated rules merely to increase arity."
        )
    elif args.proposal_profile == "trajectory_actionable":
        profile_instruction = (
            " Propose constraints that are useful while a mixed-type record is still evolving during reverse diffusion. "
            "Prioritize directional state-dependent applicability that can genuinely switch on or off as a row changes, "
            "rather than static unary bounds, demographic subgroups, rare exceptions, or duplicated summary fields. "
            "Each edge must expose a meaningful violation and at least one defensible action; prefer edges with multiple "
            "plausible endpoints when the correct repair is context dependent. Include process, temporal, accounting, "
            "cross-field magnitude, and conditional co-occurrence relations. A graph is better because its edges are "
            "semantically distinct and actionable, not because it contains more predicates."
        )
    elif args.proposal_profile == "connectivity_aware":
        profile_instruction = (
            " Use the supplied existing_graph_context to seek semantically defensible bridges between disconnected or weakly "
            "connected schema components. Prioritize executable three-or-more-column conditional hyperedges, process chains, "
            "and conjunctions whose atoms are jointly necessary. A useful bridge should let the graph coordinate a repair "
            "across related processes; do not invent an edge merely to reduce the component count. Avoid paraphrases, threshold "
            "variants, and empirical demographic or target correlations. State whether a proposed bridge should actuate, only "
            "score candidate actions, or remain diagnostic, but accept that authority is assigned downstream after validation."
        )
    elif args.proposal_profile == "semantic_programmatic":
        profile_instruction = (
            " Follow the few_shot_epistemic_examples as demonstrations of reasoning roles, not as candidates to copy. "
            "Separate three outputs: (1) documented type/domain invariants that may actuate, (2) conditional process or "
            "measurement hyperedges with value-dependent applicability, and (3) causal or external-knowledge hypotheses "
            "that remain soft scoring messages until an authoritative mechanism is available. Prefer integer/count, "
            "nonnegative quantity, subset, min/mean/max, probability, temporal, and process semantics, but encode only "
            "inter-column relations supported by this DSL. Do not waste LLM calls rediscovering unary dtype/range facts: "
            "those belong to a deterministic schema compiler. Every soft hypothesis must state a plausible omitted variable "
            "or counterexample and must not directly overwrite a row."
        )
    base_instruction = (
        "Construct executable, directional, inter-column knowledge hyperedges for synthetic tabular data. "
        "Do not emit single-column tautologies, restated bounds, target leakage, sensitive stereotypes, or symmetric correlations. "
        "Hardness is semantic: hard means almost no legitimate exceptions; near_hard permits a calibrated exception mass; soft is distributional only. "
        "Use exact schema names and listed categorical values. Support literal, column-comparison, temporal, process, and accounting relations. "
        "Prefer semantic equations, directional inequalities, conditional denial constraints, accounting identities or bounds, "
        "temporal ordering, process-coupled missingness, value-conditioned dependencies, and explicit exception conditions over "
        "mere high correlation or frequent-value rules. Every applicability, requirement, and exception atom must pass a "
        "counterfactual necessity test: removing that atom should change which rows the relation applies to, which violations it "
        "detects, or which legitimate cases are exempted. Never add an atom only to increase arity. State the external or process "
        "semantics that justify the direction; empirical support alone cannot turn a correlation into a hard constraint. "
        "Treat every codebook_caveat as a binding epistemic limit: never invent a billing-cycle alignment, categorical meaning, "
        "or omitted field. A relation that depends on unknown metadata must remain soft/diagnostic and explicitly name the missing "
        "information; it cannot receive hard or near-hard authority. "
        "For literal atoms set right_column to the empty string; for column atoms set right_value to 0 as an ignored placeholder. "
        "Raw rows are unavailable and must not be requested." + profile_instruction
    )
    call_log, usage_total, cached_calls = [], Counter(), {}
    call_path = args.output/"api_calls.jsonl"
    if args.resume and call_path.exists():
        for line in call_path.read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line); call_log.append(row)
            usage = row.get("usage", {})
            usage_total.update({k: int(v or 0) for k, v in usage.items() if isinstance(v, (int, float))})
            if not row.get("error") and row.get("result"):
                cached_calls[row["stage"]] = row

    def run(stage: str, model: str, instruction: str, payload: dict, schema_: dict, name: str, effort: str | None = None):
        invalidated = any(stage.startswith(prefix) for prefix in args.invalidate_stage_prefix)
        cached = cached_calls.get(stage)
        if cached and cached.get("model") == model and not invalidated:
            return cached["result"]
        serialized = json.dumps(native(payload), ensure_ascii=False) + json.dumps(schema_, ensure_ascii=False) + instruction
        conservative_input_estimate = math.ceil(len(serialized) / 3) + 256
        remaining = args.total_token_budget - int(usage_total.get("total_tokens", 0))
        call_maximum = min(args.max_output_tokens, remaining - conservative_input_estimate)
        if call_maximum < 16:
            progress = {"status": "budget_exhausted_before_call", "next_stage": stage,
                        "usage": dict(usage_total), "budget": args.total_token_budget, "calls": call_log}
            (args.output/"progress.json").write_text(json.dumps(native(progress), indent=2))
            raise RuntimeError(
                f"token budget prevents stage={stage}: remaining={remaining}, "
                f"estimated_input={conservative_input_estimate}"
            )
        result, usage, latency, error = retry_call(key, model, instruction, payload, schema_, name,
                                                   effort or args.reasoning_effort, maximum=call_maximum)
        call_log.append({"stage": stage, "model": model, "usage": usage, "latency": latency,
                         "max_output_tokens": call_maximum, "error": error, "result": result})
        usage_total.update({k: int(v or 0) for k, v in usage.items() if isinstance(v, (int, float))})
        with call_path.open("a") as handle:
            handle.write(json.dumps(native(call_log[-1]))+"\n")
        progress = {"status": "api_error" if error else "running", "last_stage": stage,
                    "usage": dict(usage_total), "budget": args.total_token_budget,
                    "remaining": args.total_token_budget-int(usage_total.get("total_tokens", 0)),
                    "api_calls": len(call_log)}
        (args.output/"progress.json").write_text(json.dumps(native(progress), indent=2))
        if error: raise RuntimeError(f"{stage}: {error}")
        return result

    baseline_limit = args.per_role if args.proposal_profile in {"complex_hyperedge", "trajectory_actionable", "connectivity_aware", "semantic_programmatic"} else args.per_role * 2
    baseline_payload = {**common, "empirical_signals": hints,
                        "request": f"Propose at most {baseline_limit} diverse graph edges in one pass. "
                                   f"Follow the {args.proposal_profile} profile exactly."}
    baseline_raw = run("strong_single_pass", args.external_model, base_instruction, baseline_payload,
                       PROPOSAL_SCHEMA, "single_pass_graph").get("candidates", [])

    if args.proposal_profile == "complex_hyperedge":
        roles = [
            ("process_hyperedge", args.external_model,
             "Use external process knowledge to propose multi-column state transitions, co-occurrence, temporal/accounting dependencies, and executable denial constraints. Identify all jointly necessary atoms and plausible alternative repair endpoints.", []),
            ("conditional_structure", args.proposal_model,
             "Use aggregate signals to form nonvacuous conditional dependencies and conjunctions. Prefer a simpler edge when extra atoms do not materially change applicability or counterexamples.", hints),
            ("exception_deactivation", args.critic_model,
             "Search for directional hyperedges whose validity changes under explicit row state. Represent legitimate exception or deactivation conditions directly, and distinguish them from post-hoc subgroup fitting.", hints),
        ]
    elif args.proposal_profile == "trajectory_actionable":
        roles = [
            ("stateful_process", args.external_model,
             "Use external process and measurement knowledge to propose directional constraints whose applicability can change as a partial row evolves. Require a clear executable action and avoid redundant codebook summaries.", []),
            ("actionable_structure", args.proposal_model,
             "Use aggregate evidence to propose diverse conditional relations with stable support, clear direction, and nontrivial repair choices. Prefer distinct endpoint families over many threshold variants of one relation.", hints),
            ("trajectory_ambiguity", args.critic_model,
             "Find constraints where an intermediate violation admits more than one plausible action, so row state and graph context can disambiguate. Reject niche exception-only rules and accidental subgroup correlations.", hints),
        ]
    elif args.proposal_profile == "connectivity_aware":
        roles = [
            ("semantic_bridge", args.external_model,
             "Use domain and process knowledge to propose a small number of meaningful bridges across supplied graph components. Prefer process chains or accounting relationships over statistical association, and name all necessary conditioning columns.", []),
            ("validated_hyperedge", args.proposal_model,
             "Use aggregate signals to test whether multi-column conjunctions connect endpoint families more cleanly than pairwise rules. Extra atoms must change applicability or counterexamples; never add an atom only to raise arity.", hints),
            ("coordination_guardrail", args.critic_model,
             "Propose relations useful for choosing among multiple repair endpoints: they should detect collateral inconsistency introduced by an action. Mark plausible relations as scoring guardrails when they are not universal enough to actuate.", hints),
        ]
    elif args.proposal_profile == "semantic_programmatic":
        roles = [
            ("documented_invariant", args.external_model,
             "Recover only externally defensible measurement, subset, count, summary-statistic, probability, and process invariants. Distinguish deterministic schema/type facts from inter-column graph edges and do not emit unary dtype bounds.", []),
            ("state_conditioned_process", args.proposal_model,
             "Use value-aware aggregate signals only to retrieve contexts, then propose conditional directional hyperedges whose applicability can change with row state. Every atom must be necessary and the reverse direction must be considered explicitly.", hints),
            ("soft_external_hypothesis", args.critic_model,
             "Propose nontrivial causal or process hypotheses only as soft action-scoring messages. Name omitted variables and legitimate counterexamples; reject demographic, target-related, and merely correlational validity rules.", hints),
        ]
    else:
        roles = [
            ("external_domain", args.external_model,
             "Use external domain/process knowledge. Prefer relations not obtainable from marginal statistics alone. State realistic exception/deactivation conditions.", []),
            ("empirical_structure", args.proposal_model,
             "Interpret the aggregate empirical signals as possible directional structure. Reject accidental correlations and improve weak signals with meaningful applicability.", hints),
            ("directionality_exception", args.critic_model,
             "Focus on asymmetric direction, temporal/process ordering, conjunctions, and conditions that distinguish valid dependency from reverse or subgroup artifacts.", hints),
        ]
    baseline_valid, baseline_rejected = prepare_candidates(
        baseline_raw, "strong_single_pass", discovery, counterexample, categorical, numerical,
        target, audit.is_sensitive, "base")
    role_raw = list(seed_candidates)
    role_valid, role_rejected = prepare_candidates(
        seed_candidates, "predefined_seed", discovery, counterexample, categorical, numerical,
        target, audit.is_sensitive, "seed",
    )
    previous_round_audit = None
    for round_index in range(args.rounds):
        round_raw = []
        for role, model, specialization, role_hints in roles:
            payload = {**common, "role": role, "proposal_round": round_index + 1,
                       "empirical_signals": role_hints, "previous_round_audit": previous_round_audit,
                       "request": (
                           f"Propose at most {args.per_role} nonduplicate candidates. {specialization} "
                           "In later rounds, fill missing endpoint/family coverage and address audited failure modes; "
                           "do not merely paraphrase retained candidates."
                       )}
            result = run(f"proposal_r{round_index+1}_{role}", model, base_instruction+" "+specialization, payload,
                         PROPOSAL_SCHEMA, f"graph_proposal_r{round_index+1}_{role}")
            for item in result.get("candidates", []):
                item["proposal_role"] = role
                item["proposal_round"] = round_index + 1
            round_raw.extend(result.get("candidates", []))
        round_valid, round_rejected = prepare_candidates(
            round_raw, "multi_role", discovery, counterexample, categorical, numerical,
            target, audit.is_sensitive, f"r{round_index+1}")
        role_raw.extend(round_raw); role_valid.extend(round_valid); role_rejected.extend(round_rejected)
        previous_round_audit = {
            "rejection_reason_counts": dict(Counter(x["rejection"] for x in round_rejected)),
            "surviving_candidates": [
                {"description": x["description"], "relation_family": x["relation_family"],
                 "applicability_columns": sorted(set(candidate_columns({**x, "requirements": [], "exceptions": []}))),
                 "requirement_columns": sorted(set(candidate_columns({**x, "applicability": [], "exceptions": []}))),
                 "discovery_support": x["discovery"]["support"],
                 "discovery_violation_rate": x["discovery"]["violation_rate"],
                 "counterexample_support": x["counterexample"]["support"],
                 "counterexample_violation_rate": x["counterexample"]["violation_rate"],
                 "counterexample_aggregate_signature": counterexample_signature(
                     counterexample, x, target, audit.is_sensitive
                 )}
                for x in round_valid[:30]
            ],
            "instruction": (
                "These aggregate diagnostics are counterexample feedback, not proof of semantic validity. "
                "Use a signature only to formulate a semantically defensible applicability/exception hypothesis; "
                "do not add the strongest correlating field mechanically."
            ),
        }

    # Avoid paying the critic repeatedly for exact duplicates across proposal roles/rounds.
    precritic = {}
    for item in role_valid:
        key_ = item["canonical"]
        score = min(item["discovery"]["confidence"], item["counterexample"]["confidence"])
        if key_ not in precritic or score > precritic[key_][0]:
            precritic[key_] = (score, item)
    role_valid = [value[1] for value in precritic.values()]
    if args.max_review_candidates > 0:
        role_valid = role_valid[:args.max_review_candidates]

    def compact_review_context(items: list[dict]) -> dict:
        """Send reviewers only schema entries referenced by the current candidate block."""
        referenced = {column for item in items for column in candidate_columns(item)}
        compact_schema = {
            "privacy": schema["privacy"],
            "rows": schema["rows"],
            "columns": [column for column in schema["columns"] if column["name"] in referenced],
        }
        return {
            "privacy": common["privacy"],
            "dataset": common["dataset"],
            "dataset_context": common["dataset_context"],
            "target_column": common["target_column"],
            "column_glossary": {
                key: value for key, value in common["column_glossary"].items()
                if key in referenced or any(part in referenced for part in key.split(".."))
            },
            "codebook_caveats": common["codebook_caveats"],
            "schema": compact_schema,
            "semantics": common["semantics"],
        }

    critique_instruction = (
        "Act as an adversarial database-constraint reviewer. For every candidate, inspect directionality, semantic plausibility, "
        "hardness, aggregate counterexamples, sensitive/normative risk and vacuity. Keep, revise, or reject it. "
        "Enforce codebook_caveats: unknown temporal alignment, undocumented categorical meanings, or omitted process fields cannot "
        "support hard/near-hard authority. A data-fitting exception is not valid unless its process meaning is independently defensible. "
        "For keep, copy the candidate exactly into the revised fields. Never infer that zero observed violations proves a universal law."
    )
    def critique_candidates(items: list[dict], cohort: str) -> list[dict]:
        reviewed_items = []
        for start in range(0, len(items), args.critic_batch_size):
            block = items[start:start+args.critic_batch_size]
            payload = {**compact_review_context(block), "cohort": cohort, "candidates": [
                {key: native(item[key]) for key in ["candidate_id", *CANDIDATE_REQUIRED, "discovery", "counterexample"]}
                for item in block]}
            batch_index = start // args.critic_batch_size
            result = run(f"critic_{cohort}_{batch_index}", args.critic_model, critique_instruction, payload,
                         REVIEW_SCHEMA, f"graph_critique_{cohort}_{batch_index}")
            by_id = {x["candidate_id"]: x for x in block}
            for review in result.get("reviews", []):
                original = by_id.get(review["candidate_id"])
                if original is None or review["decision"] == "reject": continue
                candidate = {key: review[key] for key in CANDIDATE_REQUIRED}
                ok, _ = typecheck(candidate, discovery, categorical, numerical, target, audit.is_sensitive)
                if not ok: continue
                try: d, x = verify(discovery, candidate), verify(counterexample, candidate)
                except Exception: continue
                reviewed_items.append({
                    **candidate, "candidate_id": original["candidate_id"],
                    "source": f"{cohort}_critic", "proposal_role": original.get("proposal_role"),
                    "proposal_round": original.get("proposal_round"),
                    "critic_decision": review["decision"], "critic_confidence": review["semantic_confidence"],
                    "critic_issues": review["issue_codes"], "critic_text": review["critique"],
                    "canonical": canonical(candidate), "discovery": d, "counterexample": x,
                })
        dedup = {}
        for item in reviewed_items:
            key_ = item["canonical"]
            if key_ not in dedup or item["critic_confidence"] > dedup[key_]["critic_confidence"]:
                dedup[key_] = item
        return list(dedup.values())

    reviewed = critique_candidates(role_valid, "multi_role")
    baseline_reviewed = critique_candidates(baseline_valid, "single_pass")

    def role_balanced_cap(items: list[dict], maximum: int) -> list[dict]:
        if maximum <= 0 or len(items) <= maximum:
            return items
        groups: dict[str, list[dict]] = {}
        for item in items:
            groups.setdefault(str(item.get("proposal_role") or item.get("source") or "unknown"), []).append(item)
        selected = []
        while len(selected) < maximum and any(groups.values()):
            for role in list(groups):
                if groups[role] and len(selected) < maximum:
                    selected.append(groups[role].pop(0))
        return selected

    reviewed = role_balanced_cap(reviewed, args.max_judge_candidates)
    baseline_reviewed = role_balanced_cap(baseline_reviewed, args.max_judge_candidates)
    judge_instruction = (
        "You are the final semantic judge for a tabular knowledge graph. Decide whether each directional edge is semantically "
        "defensible and appropriately scoped. Aggregate evidence establishes empirical fit but not semantics. Retain meaningful documented "
        "representation/codebook dependencies as dataset_semantic, but reject arbitrary identifier correlations, tautologies, sensitive norms, "
        "wrong direction, or unjustified hardness. Treat codebook_caveats as binding: reject hard/near-hard decisions that rely on unknown "
        "cycle alignment, undocumented codes, or omitted process fields. Do not revise candidates and do not see calibration/test data."
    )
    def judge_candidates(items: list[dict], cohort: str) -> list[dict]:
        judgments = {}
        for start in range(0, len(items), args.judge_batch_size):
            block = items[start:start+args.judge_batch_size]
            payload = {**compact_review_context(block), "cohort": cohort, "candidates": [
                {key: native(item[key]) for key in ["candidate_id", *CANDIDATE_REQUIRED, "discovery", "counterexample",
                                                    "critic_decision", "critic_confidence", "critic_issues", "critic_text"]}
                for item in block]}
            result = run(f"judge_{cohort}_{start//args.judge_batch_size}", args.judge_model, judge_instruction, payload,
                         JUDGE_SCHEMA, f"graph_judgment_{cohort}_{start//args.judge_batch_size}", effort=args.judge_reasoning_effort)
            judgments.update({x["candidate_id"]: x for x in result.get("judgments", [])})
        judged_items = []
        for item in items:
            judgment = judgments.get(item["candidate_id"])
            if judgment is None: continue
            judged_items.append({**item, "judge_decision": judgment["decision"],
                                 "judge_confidence": judgment["semantic_confidence"],
                                 "generalizability": judgment["generalizability"],
                                 "judge_issues": judgment["issue_codes"], "judge_rationale": judgment["rationale"]})
        return judged_items

    judged = judge_candidates(reviewed, "multi_role")
    baseline_judged = judge_candidates(baseline_reviewed, "single_pass")

    baseline_unjudged_frozen, baseline_unjudged_cal_rejected = select_after_calibration(
        baseline_valid, calibration, heldout, args.min_support, require_judge=False)
    baseline_frozen, baseline_cal_rejected = select_after_calibration(
        baseline_judged, calibration, heldout, args.min_support, require_judge=True)
    protocol_frozen, protocol_cal_rejected = select_after_calibration(
        judged, calibration, heldout, args.min_support, require_judge=True)

    # Empirical-only graph passes the same blind split logic. It is kept separate
    # so foundation-model incremental value remains measurable.
    empirical_ready = []
    for index, item in enumerate(empirical_candidates):
        empirical_ready.append({**{k: item[k] for k in CANDIDATE_REQUIRED},
                                "candidate_id": f"emp_{index:03d}", "source": "empirical_mining",
                                "canonical": canonical(item), "discovery": item["discovery"],
                                "counterexample": verify(counterexample, item)})
    empirical_frozen, empirical_rejected = select_after_calibration(
        empirical_ready, calibration, heldout, args.min_support, require_judge=False)

    merged = {}
    for item in empirical_frozen + protocol_frozen:
        key_ = item["canonical"]
        if key_ not in merged:
            merged[key_] = {**item, "provenance": [item["source"]]}
        else:
            merged[key_]["provenance"].append(item["source"])
            if item["source"] != "empirical_mining":
                # Preserve the semantically reviewed description and family.
                for field in ["description", "relation_family", "evidence_basis", "rationale",
                              "directionality_reason", "plausible_counterexample", "judge_confidence",
                              "generalizability", "judge_rationale"]:
                    if field in item: merged[key_][field] = item[field]
    unified = list(merged.values())
    summaries = [
        graph_summary("strong_single_pass_unjudged", baseline_unjudged_frozen, len(baseline_raw)),
        graph_summary("strong_single_pass", baseline_frozen, len(baseline_raw)),
        graph_summary("llm_protocol", protocol_frozen, len(role_raw)),
        graph_summary("empirical_only", empirical_frozen, len(empirical_candidates)),
        graph_summary("unified_protocol", unified, len(role_raw)+len(empirical_candidates)),
    ]

    for name, data in [("strong_single_pass_unjudged_graph", baseline_unjudged_frozen),
                       ("strong_single_pass_graph", baseline_frozen), ("llm_protocol_graph", protocol_frozen),
                       ("empirical_graph", empirical_frozen), ("unified_graph", unified)]:
        (args.output/f"{name}.json").write_text(json.dumps(native(data), indent=2))
    (args.output/"value_aware_dependency_signals.json").write_text(
        json.dumps(native(value_aware_hints), indent=2)
    )
    (args.output/"stage_rejections.json").write_text(json.dumps(native({
        "baseline_prefilter": baseline_rejected, "role_prefilter": role_rejected,
        "baseline_unjudged_calibration": baseline_unjudged_cal_rejected,
        "baseline_calibration": baseline_cal_rejected, "protocol_calibration": protocol_cal_rejected,
        "empirical_calibration": empirical_rejected}), indent=2))
    pd.DataFrame(summaries).to_csv(args.output/"graph_comparison.csv", index=False)
    metadata = {"dataset": args.dataset, "models": {"proposal": args.proposal_model,
                "external": args.external_model, "critic": args.critic_model, "judge": args.judge_model},
                "proposal_rounds": args.rounds,
                "proposal_profile": args.proposal_profile,
                "dsl": "typed_hyperedge_v2_literal_and_column_comparison",
                "protocol_version": "v4_authority_ceiling_fair_critic_judge_and_codebook_scope",
                "token_budget": args.total_token_budget, "max_output_tokens_per_call": args.max_output_tokens,
                "split_rows": {"discovery": len(discovery), "counterexample": len(counterexample),
                               "calibration": len(calibration), "heldout": len(heldout)},
                "usage": dict(usage_total), "api_calls": len(call_log),
                "value_aware_dependency_signals": len(value_aware_hints),
                "real_rows_sent": False, "heldout_used_for_selection": False,
                "target_or_sensitive_edges_allowed": False}
    (args.output/"run_metadata.json").write_text(json.dumps(native(metadata), indent=2))
    lines = ["# Multi-stage knowledge graph generation protocol", "",
             f"- Dataset: `{args.dataset}`", f"- API calls: **{len(call_log)}**",
             f"- Total tokens: **{usage_total.get('total_tokens', 0)}**", "",
             "| Graph | Proposed | Retained | Hard | Near-hard | Soft | Families | Endpoint patterns | External/hybrid | Heldout stable | Mean shift |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['graph']} | {row['proposed']} | {row['retained']} | {row['hard']} | {row['near_hard']} | {row['soft']} | "
                     f"{row['family_diversity']} | {row['endpoint_patterns']} | {row['external_or_hybrid']} | "
                     f"{row['heldout_stable_005']:.3f} | {row['mean_heldout_shift']:.4f} |")
    lines += ["", "Heldout data was opened only after semantic judgment and blind calibration froze each graph. No raw source row was sent to the API."]
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(pd.DataFrame(summaries).to_string(index=False)); print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__": main()
