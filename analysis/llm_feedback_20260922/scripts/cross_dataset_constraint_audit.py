#!/usr/bin/env python3
"""Train-only discovery audit for record-dependent tabular constraints.

The script never prints or writes source rows.  It fits predicates/mappings on
the repository's training split and uses the held-out split only to count
support and exceptions.  The resulting candidates are *empirical hypotheses*,
not automatically ground-truth semantic constraints.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


SENSITIVE_RE = re.compile(
    r"(^|[_. ])(age|sex|gender|race|ethnic|marital|relationship|native[_. ]?country)($|[_. ])",
    re.IGNORECASE,
)


@dataclass
class Predicate:
    predicate_id: str
    column: str
    description: str
    kind: str
    operator: str
    value: Any
    train_mask: np.ndarray
    test_mask: np.ndarray


def native(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def canonical_categorical(series: pd.Series) -> pd.Series:
    """Canonicalize public categorical values and join missing-like tokens."""
    result = series.astype("string").str.strip()
    missing = result.isna() | result.str.lower().isin({"", "?", "na", "n/a", "nan", "unknown"})
    return result.mask(missing, "<MISSING_LIKE>")


def wilson_interval(bad: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = bad / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, centre - radius), min(1.0, centre + radius)


def split_metrics(applicable: np.ndarray, satisfied: np.ndarray) -> dict[str, Any]:
    applicable = np.asarray(applicable, dtype=bool)
    satisfied = np.asarray(satisfied, dtype=bool)
    n_applicable = int(applicable.sum())
    violations = int(np.logical_and(applicable, ~satisfied).sum())
    rate = violations / n_applicable if n_applicable else float("nan")
    low, high = wilson_interval(violations, n_applicable)
    return {
        "applicable": n_applicable,
        "violations": violations,
        "violation_rate": rate,
        "violation_wilson95_low": low,
        "violation_wilson95_high": high,
    }


def empirical_class(violations: int, applicable: int) -> str:
    if applicable <= 0:
        return "untested"
    rate = violations / applicable
    if violations == 0:
        return "zero_observed_exceptions"
    if rate <= 0.01:
        return "near_hard_le_1pct"
    return "probabilistic"


def heldout_class(train: dict[str, Any], test: dict[str, Any]) -> str:
    if test["applicable"] == 0:
        return "heldout_not_applicable"
    tr = train["violation_rate"]
    te = test["violation_rate"]
    if train["violations"] == 0 and test["violations"] == 0:
        return "no_observed_exception_both_splits"
    if tr <= 0.01 and te <= 0.01:
        return "near_hard_stable"
    if abs(tr - te) <= 0.05:
        return "probabilistic_stable"
    return "exception_rate_shift"


def is_sensitive(columns: list[str]) -> bool:
    return any(SENSITIVE_RE.search(str(column).replace("-", "_")) for column in columns)


def annotate_candidate(
    row: dict[str, Any], columns: list[str], target: str | None, train_rows: int, test_rows: int
) -> dict[str, Any]:
    train_app = int(row["train_applicable"])
    train_bad = int(row["train_violations"])
    test_app = int(row["test_applicable"])
    test_bad = int(row["test_violations"])
    train = {
        "applicable": train_app,
        "violations": train_bad,
        "violation_rate": row["train_violation_rate"],
    }
    test = {
        "applicable": test_app,
        "violations": test_bad,
        "violation_rate": row["test_violation_rate"],
    }
    sensitive = is_sensitive(columns)
    target_related = bool(target and target in columns)
    row["train_empirical_class"] = empirical_class(train_bad, train_app)
    row["heldout_class"] = heldout_class(train, test)
    row["train_coverage"] = train_app / train_rows
    row["test_coverage"] = test_app / test_rows
    row["rare_subgroup"] = bool(row["train_coverage"] < 0.05)
    row["support_strength"] = (
        "adequate_for_pilot" if train_app >= 100 and test_app >= 30 else "insufficient_heldout_support"
    )
    row["sensitive_attribute"] = sensitive
    row["target_related"] = target_related

    if sensitive:
        row["use_category"] = "audit_only_sensitive_or_demographic"
    elif target_related:
        row["use_category"] = "distributional_only_involves_target"
    elif row["support_strength"] == "insufficient_heldout_support":
        row["use_category"] = "insufficient_support_stress_test"
    elif row["heldout_class"] == "no_observed_exception_both_splits":
        row["use_category"] = "executable_baseline_candidate"
    elif row["heldout_class"] == "near_hard_stable":
        row["use_category"] = "near_hard_dynamic_benchmark"
    elif row["heldout_class"] == "exception_rate_shift":
        row["use_category"] = "subgroup_or_shift_stress_test"
    else:
        row["use_category"] = "probabilistic_dependency_not_hard_validity"
    return row


def predicate_bank(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
    numerical: list[str],
    min_support: int,
) -> list[Predicate]:
    predicates: list[Predicate] = []
    n_train = len(train)

    for column in categorical:
        train_values = canonical_categorical(train[column])
        test_values = canonical_categorical(test[column])
        counts = train_values.value_counts(dropna=False)
        for value, support in counts.items():
            if int(support) < min_support or int(support) > n_train - min_support:
                continue
            safe_value = str(value)
            predicates.append(
                Predicate(
                    predicate_id=f"{column}=={safe_value}",
                    column=column,
                    description=f"{column} == {safe_value}",
                    kind="categorical_value",
                    operator="eq",
                    value=safe_value,
                    train_mask=train_values.eq(value).to_numpy(dtype=bool),
                    test_mask=test_values.eq(value).to_numpy(dtype=bool),
                )
            )

    for column in numerical:
        tr = pd.to_numeric(train[column], errors="coerce")
        te = pd.to_numeric(test[column], errors="coerce")
        finite = tr[np.isfinite(tr)]
        if finite.empty:
            continue

        specifications: list[tuple[str, str, float, Callable[[pd.Series, float], pd.Series]]] = [
            ("zero", "==", 0.0, lambda values, threshold: values.eq(threshold)),
            ("positive", ">", 0.0, lambda values, threshold: values.gt(threshold)),
        ]
        q10 = float(finite.quantile(0.10))
        q90 = float(finite.quantile(0.90))
        if q10 < q90:
            specifications.extend(
                [
                    ("low_tail", "<=", q10, lambda values, threshold: values.le(threshold)),
                    ("high_tail", ">=", q90, lambda values, threshold: values.ge(threshold)),
                ]
            )

        seen: set[tuple[str, float]] = set()
        for kind, operator, threshold, apply in specifications:
            key = (operator, round(threshold, 12))
            if key in seen:
                continue
            seen.add(key)
            train_mask = apply(tr, threshold).fillna(False).to_numpy(dtype=bool)
            support = int(train_mask.sum())
            if support < min_support or support > n_train - min_support:
                continue
            test_mask = apply(te, threshold).fillna(False).to_numpy(dtype=bool)
            formatted = f"{threshold:.8g}"
            predicates.append(
                Predicate(
                    predicate_id=f"{column}{operator}{formatted}",
                    column=column,
                    description=f"{column} {operator} {formatted}",
                    kind=f"numeric_{kind}",
                    operator=operator,
                    value=threshold,
                    train_mask=train_mask,
                    test_mask=test_mask,
                )
            )

    return predicates


def mine_implications(
    dataset: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predicates: list[Predicate],
    target: str | None,
    min_confidence: float,
    max_per_antecedent: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_train = len(train)
    n_test = len(test)
    train_prevalence = {p.predicate_id: float(p.train_mask.mean()) for p in predicates}

    for antecedent in predicates:
        candidates: list[dict[str, Any]] = []
        for consequent in predicates:
            if antecedent.column == consequent.column:
                continue
            train_metrics = split_metrics(antecedent.train_mask, consequent.train_mask)
            confidence = 1.0 - train_metrics["violation_rate"]
            if not np.isfinite(confidence) or confidence < min_confidence:
                continue
            base_rate = train_prevalence[consequent.predicate_id]
            lift = confidence / base_rate if base_rate > 0 else float("inf")
            # Avoid rediscovering globally common consequents unless the relation is near-hard.
            if lift < 1.10 and train_metrics["violation_rate"] > 0.01:
                continue
            test_metrics = split_metrics(antecedent.test_mask, consequent.test_mask)
            row = {
                "dataset": dataset,
                "source": "train_only_predicate_mining",
                "constraint_id": f"{dataset}:{antecedent.predicate_id}->{consequent.predicate_id}",
                "constraint": f"IF {antecedent.description} THEN {consequent.description}",
                "antecedent_column": antecedent.column,
                "antecedent_kind": antecedent.kind,
                "consequent_column": consequent.column,
                "consequent_kind": consequent.kind,
                "train_applicable": train_metrics["applicable"],
                "train_violations": train_metrics["violations"],
                "train_violation_rate": train_metrics["violation_rate"],
                "train_violation_wilson95_high": train_metrics["violation_wilson95_high"],
                "test_applicable": test_metrics["applicable"],
                "test_violations": test_metrics["violations"],
                "test_violation_rate": test_metrics["violation_rate"],
                "test_violation_wilson95_high": test_metrics["violation_wilson95_high"],
                "train_confidence": confidence,
                "train_consequent_base_rate": base_rate,
                "train_lift": lift,
            }
            candidates.append(
                annotate_candidate(
                    row,
                    [antecedent.column, consequent.column],
                    target,
                    n_train,
                    n_test,
                )
            )

        # Retain a diverse but bounded set per applicability condition.
        candidates.sort(
            key=lambda r: (
                r["train_violations"] > 0,
                r["train_violation_rate"],
                -min(r["train_lift"], 100.0),
                -r["train_applicable"],
            )
        )
        rows.extend(candidates[:max_per_antecedent])
    return rows


def discover_mapping_rules(
    dataset: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
    numerical: list[str],
    target: str | None,
    max_cardinality: int,
) -> list[dict[str, Any]]:
    """Fit value-conditioned modal maps A -> B on train and audit on test."""
    result: list[dict[str, Any]] = []
    lhs_columns = [c for c in categorical if train[c].nunique(dropna=False) <= max_cardinality]
    rhs_columns = list(lhs_columns)
    rhs_columns.extend(c for c in numerical if train[c].nunique(dropna=False) <= max_cardinality)

    canonical: dict[str, tuple[pd.Series, pd.Series]] = {}
    for column in set(lhs_columns + rhs_columns):
        if column in categorical:
            canonical[column] = (canonical_categorical(train[column]), canonical_categorical(test[column]))
        else:
            canonical[column] = (
                pd.to_numeric(train[column], errors="coerce"),
                pd.to_numeric(test[column], errors="coerce"),
            )

    for lhs in lhs_columns:
        lhs_train, lhs_test = canonical[lhs]
        for rhs in rhs_columns:
            if lhs == rhs:
                continue
            rhs_train, rhs_test = canonical[rhs]
            paired = pd.DataFrame({"lhs": lhs_train, "rhs": rhs_train})
            mapping = paired.groupby("lhs", dropna=False)["rhs"].agg(
                lambda series: series.value_counts(dropna=False).index[0]
            )
            expected_train = lhs_train.map(mapping)
            expected_test = lhs_test.map(mapping)
            app_train = expected_train.notna().to_numpy(dtype=bool)
            app_test = expected_test.notna().to_numpy(dtype=bool)
            if rhs in categorical:
                sat_train = rhs_train.eq(expected_train).fillna(False).to_numpy(dtype=bool)
                sat_test = rhs_test.eq(expected_test).fillna(False).to_numpy(dtype=bool)
            else:
                sat_train = np.isclose(
                    pd.to_numeric(rhs_train, errors="coerce"),
                    pd.to_numeric(expected_train, errors="coerce"),
                    equal_nan=False,
                )
                sat_test = np.isclose(
                    pd.to_numeric(rhs_test, errors="coerce"),
                    pd.to_numeric(expected_test, errors="coerce"),
                    equal_nan=False,
                )
            train_metrics = split_metrics(app_train, sat_train)
            if train_metrics["violation_rate"] > 0.20:
                continue
            test_metrics = split_metrics(app_test, sat_test)
            row = {
                "dataset": dataset,
                "source": "train_only_modal_mapping",
                "constraint_id": f"{dataset}:MAP({lhs})->{rhs}",
                "constraint": f"{lhs} selects the train-modal value of {rhs}",
                "antecedent_column": lhs,
                "antecedent_kind": "categorical_mapping_key",
                "consequent_column": rhs,
                "consequent_kind": "categorical_value" if rhs in categorical else "low_cardinality_numeric",
                "train_applicable": train_metrics["applicable"],
                "train_violations": train_metrics["violations"],
                "train_violation_rate": train_metrics["violation_rate"],
                "train_violation_wilson95_high": train_metrics["violation_wilson95_high"],
                "test_applicable": test_metrics["applicable"],
                "test_violations": test_metrics["violations"],
                "test_violation_rate": test_metrics["violation_rate"],
                "test_violation_wilson95_high": test_metrics["violation_wilson95_high"],
                "train_confidence": 1.0 - train_metrics["violation_rate"],
                "train_consequent_base_rate": float("nan"),
                "train_lift": float("nan"),
                "unseen_test_key_count": int((~app_test).sum()),
                "mapping_cardinality": int(len(mapping)),
            }
            result.append(annotate_candidate(row, [lhs, rhs], target, len(train), len(test)))
    return result


def manual_audits(
    dataset: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str | None,
) -> list[dict[str, Any]]:
    """Small pre-specified interpretable library; no thresholds are tuned on test."""
    specs: list[tuple[str, str, list[str], Callable[[pd.DataFrame], tuple[pd.Series, pd.Series]], str]] = []

    if dataset == "adult":
        def text(column: str, value: str) -> Callable[[pd.DataFrame], pd.Series]:
            return lambda frame: canonical_categorical(frame[column]).eq(value)

        specs.extend(
            [
                (
                    "spouse_marital",
                    "relationship in {Husband, Wife} -> marital.status == Married-civ-spouse",
                    ["relationship", "marital.status"],
                    lambda frame: (
                        canonical_categorical(frame["relationship"]).isin(["Husband", "Wife"]),
                        text("marital.status", "Married-civ-spouse")(frame),
                    ),
                    "semantic",
                ),
                (
                    "husband_male",
                    "relationship == Husband -> sex == Male",
                    ["relationship", "sex"],
                    lambda frame: (text("relationship", "Husband")(frame), text("sex", "Male")(frame)),
                    "sensitive_stress_test",
                ),
                (
                    "wife_female",
                    "relationship == Wife -> sex == Female",
                    ["relationship", "sex"],
                    lambda frame: (text("relationship", "Wife")(frame), text("sex", "Female")(frame)),
                    "sensitive_stress_test",
                ),
                (
                    "own_child_age_lt30",
                    "relationship == Own-child -> age < 30",
                    ["relationship", "age"],
                    lambda frame: (
                        text("relationship", "Own-child")(frame),
                        pd.to_numeric(frame["age"], errors="coerce").lt(30),
                    ),
                    "unsafe_soft_rule_with_expected_exceptions",
                ),
                (
                    "missing_workclass_missing_occupation",
                    "workclass missing-like -> occupation missing-like",
                    ["workclass", "occupation"],
                    lambda frame: (
                        text("workclass", "<MISSING_LIKE>")(frame),
                        text("occupation", "<MISSING_LIKE>")(frame),
                    ),
                    "missingness_dependency",
                ),
            ]
        )

    elif dataset == "shoppers":
        for count, duration in (
            ("Administrative", "Administrative_Duration"),
            ("Informational", "Informational_Duration"),
            ("ProductRelated", "ProductRelated_Duration"),
        ):
            specs.extend(
                [
                    (
                        f"{count.lower()}_zero_implies_duration_zero",
                        f"{count} == 0 -> {duration} == 0",
                        [count, duration],
                        lambda frame, count=count, duration=duration: (
                            pd.to_numeric(frame[count], errors="coerce").eq(0),
                            pd.to_numeric(frame[duration], errors="coerce").eq(0),
                        ),
                        "mechanical_zero_coupling",
                    ),
                    (
                        f"{duration.lower()}_zero_implies_count_zero",
                        f"{duration} == 0 -> {count} == 0",
                        [duration, count],
                        lambda frame, count=count, duration=duration: (
                            pd.to_numeric(frame[duration], errors="coerce").eq(0),
                            pd.to_numeric(frame[count], errors="coerce").eq(0),
                        ),
                        "mechanical_zero_coupling",
                    ),
                ]
            )
        specs.extend(
            [
                (
                    "bounce_le_exit",
                    "BounceRates <= ExitRates",
                    ["BounceRates", "ExitRates"],
                    lambda frame: (
                        pd.Series(True, index=frame.index),
                        pd.to_numeric(frame["BounceRates"], errors="coerce").le(
                            pd.to_numeric(frame["ExitRates"], errors="coerce")
                        ),
                    ),
                    "cross_column_numeric",
                ),
                (
                    "pagevalue_positive_implies_revenue",
                    "PageValues > 0 -> Revenue == True",
                    ["PageValues", "Revenue"],
                    lambda frame: (
                        pd.to_numeric(frame["PageValues"], errors="coerce").gt(0),
                        canonical_categorical(frame["Revenue"]).str.lower().eq("true"),
                    ),
                    "probabilistic_outcome_dependency",
                ),
            ]
        )

    elif dataset == "default":
        specs.extend(
            [
                (
                    "positive_payment_after_zero_bill1",
                    "BILL_AMT1 <= 0 -> PAY_AMT1 == 0",
                    ["BILL_AMT1", "PAY_AMT1"],
                    lambda frame: (
                        pd.to_numeric(frame["BILL_AMT1"], errors="coerce").le(0),
                        pd.to_numeric(frame["PAY_AMT1"], errors="coerce").eq(0),
                    ),
                    "probabilistic_financial_dependency",
                ),
                (
                    "pay_status_temporal_stability",
                    "PAY_2 <= 0 -> PAY_3 <= 0",
                    ["PAY_2", "PAY_3"],
                    lambda frame: (
                        pd.to_numeric(frame["PAY_2"], errors="coerce").le(0),
                        pd.to_numeric(frame["PAY_3"], errors="coerce").le(0),
                    ),
                    "probabilistic_temporal_dependency",
                ),
            ]
        )

    elif dataset == "diabetes":
        medications = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
            "acarbose", "miglitol", "tolazamide", "insulin", "glyburide-metformin",
        ]

        def medication_values(frame: pd.DataFrame) -> pd.DataFrame:
            return frame[medications].apply(canonical_categorical)

        specs.extend(
            [
                (
                    "dose_change_implies_change_summary",
                    "any retained diabetes medication in {Up, Down} -> change == Ch",
                    medications + ["change"],
                    lambda frame: (
                        medication_values(frame).isin(["Up", "Down"]).any(axis=1),
                        canonical_categorical(frame["change"]).eq("Ch"),
                    ),
                    "data_dictionary_summary_field",
                ),
                (
                    "no_change_summary_implies_no_dose_change",
                    "change == No -> no retained diabetes medication is Up or Down",
                    ["change"] + medications,
                    lambda frame: (
                        canonical_categorical(frame["change"]).eq("No"),
                        ~medication_values(frame).isin(["Up", "Down"]).any(axis=1),
                    ),
                    "data_dictionary_summary_field",
                ),
                (
                    "active_medication_implies_diabetes_med",
                    "any retained diabetes medication != No -> diabetesMed == Yes",
                    medications + ["diabetesMed"],
                    lambda frame: (
                        medication_values(frame).ne("No").any(axis=1),
                        canonical_categorical(frame["diabetesMed"]).eq("Yes"),
                    ),
                    "data_dictionary_summary_field",
                ),
                (
                    "no_diabetes_med_implies_no_retained_medication",
                    "diabetesMed == No -> every retained diabetes medication == No",
                    ["diabetesMed"] + medications,
                    lambda frame: (
                        canonical_categorical(frame["diabetesMed"]).eq("No"),
                        medication_values(frame).eq("No").all(axis=1),
                    ),
                    "data_dictionary_summary_field",
                ),
            ]
        )

    elif dataset == "news":
        always = lambda frame: pd.Series(True, index=frame.index)
        numeric = lambda frame, column: pd.to_numeric(frame[column], errors="coerce")
        ordered = lambda frame, low, avg, high: (
            numeric(frame, low).le(numeric(frame, avg)) & numeric(frame, avg).le(numeric(frame, high))
        )
        specs.extend(
            [
                (
                    "zero_content_zero_content_rates",
                    "n_tokens_content == 0 -> token rates and average_token_length are zero",
                    [" n_tokens_content", " n_unique_tokens", " n_non_stop_words", " n_non_stop_unique_tokens", " average_token_length"],
                    lambda frame: (
                        numeric(frame, " n_tokens_content").eq(0),
                        frame[[" n_unique_tokens", " n_non_stop_words", " n_non_stop_unique_tokens", " average_token_length"]]
                        .apply(pd.to_numeric, errors="coerce").eq(0).all(axis=1),
                    ),
                    "data_dictionary_zero_denominator_identity",
                ),
                (
                    "self_reference_min_avg_max",
                    "self_reference_min_shares <= self_reference_avg_shares <= self_reference_max_shares",
                    [" self_reference_min_shares", " self_reference_avg_sharess", " self_reference_max_shares"],
                    lambda frame: (always(frame), ordered(frame, " self_reference_min_shares", " self_reference_avg_sharess", " self_reference_max_shares")),
                    "data_dictionary_order_statistic",
                ),
                (
                    "positive_polarity_min_avg_max",
                    "min_positive_polarity <= avg_positive_polarity <= max_positive_polarity",
                    [" min_positive_polarity", " avg_positive_polarity", " max_positive_polarity"],
                    lambda frame: (always(frame), ordered(frame, " min_positive_polarity", " avg_positive_polarity", " max_positive_polarity")),
                    "data_dictionary_order_statistic",
                ),
                (
                    "negative_polarity_min_avg_max",
                    "min_negative_polarity <= avg_negative_polarity <= max_negative_polarity",
                    [" min_negative_polarity", " avg_negative_polarity", " max_negative_polarity"],
                    lambda frame: (always(frame), ordered(frame, " min_negative_polarity", " avg_negative_polarity", " max_negative_polarity")),
                    "data_dictionary_order_statistic",
                ),
                (
                    "keyword_min_shares_order",
                    "kw_min_min <= kw_avg_min <= kw_max_min",
                    [" kw_min_min", " kw_avg_min", " kw_max_min"],
                    lambda frame: (always(frame), ordered(frame, " kw_min_min", " kw_avg_min", " kw_max_min")),
                    "data_dictionary_order_statistic",
                ),
                (
                    "keyword_max_shares_order",
                    "kw_min_max <= kw_avg_max <= kw_max_max",
                    [" kw_min_max", " kw_avg_max", " kw_max_max"],
                    lambda frame: (always(frame), ordered(frame, " kw_min_max", " kw_avg_max", " kw_max_max")),
                    "data_dictionary_order_statistic",
                ),
                (
                    "keyword_average_shares_order",
                    "kw_min_avg <= kw_avg_avg <= kw_max_avg",
                    [" kw_min_avg", " kw_avg_avg", " kw_max_avg"],
                    lambda frame: (always(frame), ordered(frame, " kw_min_avg", " kw_avg_avg", " kw_max_avg")),
                    "data_dictionary_order_statistic",
                ),
                (
                    "absolute_title_sentiment",
                    "abs_title_sentiment_polarity == abs(title_sentiment_polarity)",
                    [" title_sentiment_polarity", " abs_title_sentiment_polarity"],
                    lambda frame: (
                        always(frame),
                        pd.Series(np.isclose(
                            numeric(frame, " abs_title_sentiment_polarity"),
                            numeric(frame, " title_sentiment_polarity").abs(), atol=1e-10,
                        ), index=frame.index),
                    ),
                    "data_dictionary_derived_feature",
                ),
                (
                    "absolute_title_subjectivity",
                    "abs_title_subjectivity == abs(title_subjectivity - 0.5)",
                    [" title_subjectivity", " abs_title_subjectivity"],
                    lambda frame: (
                        always(frame),
                        pd.Series(np.isclose(
                            numeric(frame, " abs_title_subjectivity"),
                            (numeric(frame, " title_subjectivity") - 0.5).abs(), atol=1e-10,
                        ), index=frame.index),
                    ),
                    "data_dictionary_derived_feature",
                ),
            ]
        )

    rows: list[dict[str, Any]] = []
    for rule_id, description, columns, evaluator, family in specs:
        train_app, train_sat = evaluator(train)
        test_app, test_sat = evaluator(test)
        train_metrics = split_metrics(train_app.to_numpy(dtype=bool), train_sat.to_numpy(dtype=bool))
        test_metrics = split_metrics(test_app.to_numpy(dtype=bool), test_sat.to_numpy(dtype=bool))
        documented = dataset in {"diabetes", "news"}
        row = {
            "dataset": dataset,
            "source": "data_dictionary_derived_audit" if documented else "pre_registered_interpretable_audit",
            "provenance_url": (
                "https://archive.ics.uci.edu/dataset/296/diabetes" if dataset == "diabetes" else
                "https://archive.ics.uci.edu/dataset/332/online+news+popularity" if dataset == "news" else ""
            ),
            "constraint_id": f"{dataset}:MANUAL:{rule_id}",
            "constraint": description,
            "family": family,
            "antecedent_column": columns[0],
            "antecedent_kind": "interpretable_condition",
            "consequent_column": columns[-1],
            "consequent_kind": "interpretable_relation",
            "train_applicable": train_metrics["applicable"],
            "train_violations": train_metrics["violations"],
            "train_violation_rate": train_metrics["violation_rate"],
            "train_violation_wilson95_high": train_metrics["violation_wilson95_high"],
            "test_applicable": test_metrics["applicable"],
            "test_violations": test_metrics["violations"],
            "test_violation_rate": test_metrics["violation_rate"],
            "test_violation_wilson95_high": test_metrics["violation_wilson95_high"],
            "train_confidence": 1.0 - train_metrics["violation_rate"],
            "train_consequent_base_rate": float("nan"),
            "train_lift": float("nan"),
        }
        rows.append(annotate_candidate(row, columns, target, len(train), len(test)))
    return rows


def load_dataset(repo: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], str | None]:
    info_path = repo / "data" / dataset / "info.json"
    train_path = repo / "data" / dataset / "train.csv"
    test_path = repo / "data" / dataset / "test.csv"
    if not (info_path.exists() and train_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"missing processed split for {dataset}")
    info = json.loads(info_path.read_text())
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    columns = list(train.columns)
    if columns != list(test.columns):
        raise ValueError(f"column mismatch for {dataset}")
    numerical = [columns[int(i)] for i in info.get("num_col_idx", [])]
    categorical = [columns[int(i)] for i in info.get("cat_col_idx", [])]
    target_indices = [int(i) for i in info.get("target_col_idx", [])]
    target = columns[target_indices[0]] if target_indices else None
    if target is not None:
        if info.get("task_type") == "regression":
            numerical.append(target)
        else:
            categorical.append(target)
    return train, test, categorical, numerical, target


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: native(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def write_report(output: Path, dataset_summary: pd.DataFrame, candidates: pd.DataFrame) -> None:
    lines = [
        "# Cross-dataset dynamic-constraint feasibility audit",
        "",
        "All relations were fitted/discovered from the training split. The held-out split was used only to count applicability and exceptions. No source rows are emitted.",
        "",
        "`zero_observed_exceptions` is empirical, not proof of a universal invariant. Wilson upper bounds are included to make this distinction explicit.",
        "",
        "## Dataset summary",
        "",
        "| dataset | train | test | predicates | implications | mappings | manual audits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in dataset_summary.iterrows():
        lines.append(
            f"| {row.dataset} | {row.train_rows} | {row.test_rows} | {row.predicates} | "
            f"{row.implications} | {row.mappings} | {row.manual_audits} |"
        )

    lines.extend(["", "## Candidate use categories", ""])
    counts = candidates.groupby(["dataset", "use_category"], dropna=False).size().reset_index(name="count")
    lines.extend(["| dataset | category | count |", "|---|---|---:|"])
    for _, row in counts.iterrows():
        lines.append(f"| {row.dataset} | {row.use_category} | {row['count']} |")

    lines.extend(["", "## Interpretable audited constraints", ""])
    lines.extend(
        [
            "| dataset | constraint | train bad/app | test bad/app | held-out assessment | use |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    manual = candidates[candidates["source"].isin([
        "pre_registered_interpretable_audit", "data_dictionary_derived_audit"
    ])]
    for _, row in manual.iterrows():
        lines.append(
            f"| {row.dataset} | {row.constraint} | {row.train_violations}/{row.train_applicable} "
            f"({row.train_violation_rate:.3%}) | {row.test_violations}/{row.test_applicable} "
            f"({row.test_violation_rate:.3%}) | {row.heldout_class} | {row.use_category} |"
        )

    safe = candidates[
        candidates["use_category"].isin(["executable_baseline_candidate", "near_hard_dynamic_benchmark"])
    ].copy()
    safe = safe.sort_values(
        ["dataset", "test_violation_rate", "train_coverage"], ascending=[True, True, False]
    ).head(30)
    lines.extend(["", "## Safe shortlist (top 30)", ""])
    lines.extend(
        [
            "| dataset | constraint | train support/rate | test support/rate | use |",
            "|---|---|---:|---:|---|",
        ]
    )
    for _, row in safe.iterrows():
        lines.append(
            f"| {row.dataset} | {row.constraint} | {row.train_applicable} / {row.train_violation_rate:.3%} | "
            f"{row.test_applicable} / {row.test_violation_rate:.3%} | {row.use_category} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Any candidate touching age, sex, race, marital status, relationship, or native country is audit-only; it must not be enforced as validity.",
            "- Any candidate involving the prediction target is a distributional association, not a semantic invariant.",
            "- Candidates with fewer than 100 train-applicable or 30 test-applicable rows are marked insufficient-support even when no exception was observed.",
            "- Probabilistic and shifted rules are useful exception/subgroup stress tests. Turning them into hard guidance risks erasing legitimate modes.",
            "- The executable shortlist is suitable for rule/oracle baselines. LLM value should be tested on held-out constraint wording, applicability, and ambiguous exceptions rather than rediscovering these rules.",
            "",
        ]
    )
    (output / "AUDIT_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["adult", "default", "shoppers"])
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--max-per-antecedent", type=int, default=8)
    parser.add_argument("--mapping-max-cardinality", type=int, default=50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for dataset in args.datasets:
        try:
            train, test, categorical, numerical, target = load_dataset(args.repo, dataset)
            predicates = predicate_bank(train, test, categorical, numerical, args.min_support)
            implications = mine_implications(
                dataset,
                train,
                test,
                predicates,
                target,
                args.min_confidence,
                args.max_per_antecedent,
            )
            mappings = discover_mapping_rules(
                dataset,
                train,
                test,
                categorical,
                numerical,
                target,
                args.mapping_max_cardinality,
            )
            manual = manual_audits(dataset, train, test, target)
            all_candidates.extend(implications + mappings + manual)
            summaries.append(
                {
                    "dataset": dataset,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "categorical_columns": len(categorical),
                    "numerical_columns": len(numerical),
                    "target": target,
                    "predicates": len(predicates),
                    "implications": len(implications),
                    "mappings": len(mappings),
                    "manual_audits": len(manual),
                }
            )
        except Exception as exc:
            errors.append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})

    candidate_frame = pd.DataFrame(all_candidates)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output / "dataset_summary.csv", index=False)
    candidate_frame.to_csv(args.output / "all_constraint_candidates.csv", index=False)

    if not candidate_frame.empty:
        executable = candidate_frame[
            candidate_frame["use_category"].isin(
                ["executable_baseline_candidate", "near_hard_dynamic_benchmark"]
            )
        ].copy()
        executable.to_csv(args.output / "executable_shortlist.csv", index=False)
        stress = candidate_frame[
            candidate_frame["use_category"].isin(
                [
                    "audit_only_sensitive_or_demographic",
                    "distributional_only_involves_target",
                    "subgroup_or_shift_stress_test",
                    "insufficient_support_stress_test",
                ]
            )
        ].copy()
        stress.to_csv(args.output / "exception_and_safety_stress_tests.csv", index=False)
        write_report(args.output, summary_frame, candidate_frame)

    run_summary = {
        "datasets": summaries,
        "errors": errors,
        "total_candidates": len(candidate_frame),
        "counts_by_use_category": (
            {str(key): int(value) for key, value in candidate_frame["use_category"].value_counts().items()}
            if not candidate_frame.empty
            else {}
        ),
        "configuration": {
            "min_support": args.min_support,
            "min_confidence": args.min_confidence,
            "max_per_antecedent": args.max_per_antecedent,
            "mapping_max_cardinality": args.mapping_max_cardinality,
            "test_used_for_discovery": False,
            "raw_rows_written": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (args.output / "run_summary.json").write_text(json.dumps(run_summary, indent=2, default=native))
    print(json.dumps(run_summary, indent=2, default=native))


if __name__ == "__main__":
    main()
