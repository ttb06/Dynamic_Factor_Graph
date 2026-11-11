"""Typed graph compilation, evidence calibration, and role assignment.

Proposal dictionaries can be supplied by any source; no API credential is needed.
"""
from __future__ import annotations

import math


import numpy as np


import pandas as pd


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
