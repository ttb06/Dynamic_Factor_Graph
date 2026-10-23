#!/usr/bin/env python3
"""Profile mixed-type datasets and audit train-only rules without emitting rows.

All candidate relationships are fitted on the training split.  The test split is
used only for reporting generalisation/exception rates.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def violation_summary(mask: pd.Series, applicable: pd.Series) -> tuple[int, int, float]:
    applicable = applicable.fillna(False).astype(bool)
    mask = mask.fillna(False).astype(bool) & applicable
    n_app = int(applicable.sum())
    n_bad = int(mask.sum())
    return n_app, n_bad, float(n_bad / n_app) if n_app else float("nan")


def add_rule(rows, dataset, rule_id, family, status, description, train_pair, test_pair, note=""):
    rows.append(
        {
            "dataset": dataset,
            "rule_id": rule_id,
            "family": family,
            "status": status,
            "description": description,
            "train_applicable": train_pair[0],
            "train_violations": train_pair[1],
            "train_violation_rate": train_pair[2],
            "test_applicable": test_pair[0],
            "test_violations": test_pair[1],
            "test_violation_rate": test_pair[2],
            "note": note,
        }
    )


def audit_manual_rules(name: str, train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if name == "adult":
        # The mapping is learned exclusively from train.  It is not hard-coded from test rows.
        edu_map = (
            train.groupby("education", dropna=False)["education.num"]
            .agg(lambda s: s.value_counts(dropna=False).index[0])
            .to_dict()
        )
        for split, df in (("train", train), ("test", test)):
            expected = df["education"].map(edu_map)
            app = expected.notna()
            bad = ~np.isclose(df["education.num"].astype(float), expected.astype(float), equal_nan=False)
            if split == "train":
                train_pair = violation_summary(pd.Series(bad, index=df.index), app)
            else:
                test_pair = violation_summary(pd.Series(bad, index=df.index), app)
        add_rule(
            rows,
            name,
            "adult.education_fd",
            "functional_dependency",
            "hard_candidate",
            "education determines education.num using the train-only modal mapping",
            train_pair,
            test_pair,
            "Executable and exactly repairable; an LLM is not needed once discovered.",
        )

        rules = [
            ("adult.husband_male", "relationship == Husband implies sex == Male", "Husband", "Male"),
            ("adult.wife_female", "relationship == Wife implies sex == Female", "Wife", "Female"),
        ]
        for rid, desc, relation, sex in rules:
            pairs = []
            for df in (train, test):
                app = df["relationship"].astype(str).str.strip().eq(relation)
                bad = ~df["sex"].astype(str).str.strip().eq(sex)
                pairs.append(violation_summary(bad, app))
            add_rule(rows, name, rid, "conditional_semantic", "hard_candidate", desc, pairs[0], pairs[1])

        pairs = []
        for df in (train, test):
            app = df["relationship"].astype(str).str.strip().isin(["Husband", "Wife"])
            bad = ~df["marital.status"].astype(str).str.strip().eq("Married-civ-spouse")
            pairs.append(violation_summary(bad, app))
        add_rule(
            rows,
            name,
            "adult.spouse_marital",
            "conditional_semantic",
            "hard_candidate",
            "relationship in {Husband, Wife} implies marital.status == Married-civ-spouse",
            pairs[0],
            pairs[1],
        )

        for col in ("capital.gain", "capital.loss", "hours.per.week"):
            pairs = []
            for df in (train, test):
                app = pd.Series(True, index=df.index)
                lower = 1 if col == "hours.per.week" else 0
                bad = pd.to_numeric(df[col], errors="coerce").lt(lower)
                pairs.append(violation_summary(bad, app))
            add_rule(
                rows,
                name,
                f"adult.{col}_nonnegative",
                "range",
                "hard_candidate",
                f"{col} >= {1 if col == 'hours.per.week' else 0}",
                pairs[0],
                pairs[1],
            )

        # Deliberately audit a tempting but unsafe subgroup rule to expose rare valid exceptions.
        pairs = []
        for df in (train, test):
            app = df["relationship"].astype(str).str.strip().eq("Own-child")
            bad = pd.to_numeric(df["age"], errors="coerce").ge(30)
            pairs.append(violation_summary(bad, app))
        add_rule(
            rows,
            name,
            "adult.own_child_age_lt30",
            "subgroup_probabilistic",
            "unsafe_soft_rule",
            "relationship == Own-child suggests age < 30",
            pairs[0],
            pairs[1],
            "Counterexample audit: enforcing this as hard can erase a legitimate tail subgroup.",
        )

    elif name == "default":
        normalized = {str(c).strip().lower().replace(" ", "_"): c for c in train.columns}
        limit_col = next((normalized[k] for k in normalized if "limit_bal" in k), None)
        age_col = next((normalized[k] for k in normalized if k == "age"), None)
        for rid, col, lower, desc in (
            ("default.limit_positive", limit_col, 1, "credit limit is positive"),
            ("default.age_adult", age_col, 18, "age is at least 18"),
        ):
            if col is None:
                continue
            pairs = []
            for df in (train, test):
                app = pd.Series(True, index=df.index)
                bad = pd.to_numeric(df[col], errors="coerce").lt(lower)
                pairs.append(violation_summary(bad, app))
            add_rule(rows, name, rid, "range", "hard_candidate", desc, pairs[0], pairs[1])

    elif name == "shoppers":
        for col in [c for c in train.columns if str(c).lower() in {"administrative", "informational", "productrelated"}]:
            pairs = []
            for df in (train, test):
                app = pd.Series(True, index=df.index)
                values = pd.to_numeric(df[col], errors="coerce")
                bad = values.lt(0) | ~np.isclose(values, np.round(values), equal_nan=False)
                pairs.append(violation_summary(pd.Series(bad, index=df.index), app))
            add_rule(
                rows,
                name,
                f"shoppers.{col}_count",
                "type_range",
                "hard_candidate",
                f"{col} is a non-negative integer count",
                pairs[0],
                pairs[1],
            )
        for col in [c for c in train.columns if str(c).lower() in {"bouncerates", "exitrates", "pagevalues"}]:
            pairs = []
            for df in (train, test):
                app = pd.Series(True, index=df.index)
                values = pd.to_numeric(df[col], errors="coerce")
                upper = 1 if str(col).lower() != "pagevalues" else np.inf
                bad = values.lt(0) | values.gt(upper)
                pairs.append(violation_summary(pd.Series(bad, index=df.index), app))
            add_rule(rows, name, f"shoppers.{col}_range", "range", "hard_candidate", f"valid range for {col}", pairs[0], pairs[1])
    return rows


def discover_train_only_rules(name: str, train: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> list[dict]:
    """Find compact deterministic/probabilistic relations; do not inspect test when fitting."""
    rows: list[dict] = []
    max_cardinality = 50
    min_support = max(20, int(0.001 * len(train)))

    eligible_cat = [c for c in cat_cols if train[c].nunique(dropna=False) <= max_cardinality]
    for lhs in eligible_cat:
        for rhs in eligible_cat:
            if lhs == rhs:
                continue
            grouped = train.groupby(lhs, dropna=False)[rhs]
            for lhs_value, values in grouped:
                if len(values) < min_support:
                    continue
                counts = values.value_counts(dropna=False)
                predicted = counts.index[0]
                confidence = float(counts.iloc[0] / len(values))
                if confidence < 0.98:
                    continue
                train_app = train[lhs].fillna("<NA>").astype(str).eq("<NA>" if pd.isna(lhs_value) else str(lhs_value))
                test_app = test[lhs].fillna("<NA>").astype(str).eq("<NA>" if pd.isna(lhs_value) else str(lhs_value))
                train_bad = ~train[rhs].fillna("<NA>").astype(str).eq("<NA>" if pd.isna(predicted) else str(predicted))
                test_bad = ~test[rhs].fillna("<NA>").astype(str).eq("<NA>" if pd.isna(predicted) else str(predicted))
                rows.append(
                    {
                        "dataset": name,
                        "lhs_column": lhs,
                        "lhs_value": str(lhs_value),
                        "rhs_column": rhs,
                        "predicted_value": str(predicted),
                        "support": int(len(values)),
                        "train_confidence": confidence,
                        "test_applicable": int(test_app.sum()),
                        "test_violation_rate": violation_summary(test_bad, test_app)[2],
                        "candidate_kind": "exact" if confidence == 1.0 else "probabilistic",
                    }
                )

    # Discover categorical -> low-cardinality numerical functional dependencies.
    low_card_num = [c for c in num_cols if train[c].nunique(dropna=False) <= 30]
    for lhs in eligible_cat:
        for rhs in low_card_num:
            mapping = train.groupby(lhs, dropna=False)[rhs].agg(lambda s: s.value_counts(dropna=False).index[0])
            expected_train = train[lhs].map(mapping)
            expected_test = test[lhs].map(mapping)
            train_app = expected_train.notna()
            test_app = expected_test.notna()
            train_bad = ~np.isclose(pd.to_numeric(train[rhs]), pd.to_numeric(expected_train), equal_nan=False)
            test_bad = ~np.isclose(pd.to_numeric(test[rhs]), pd.to_numeric(expected_test), equal_nan=False)
            train_pair = violation_summary(pd.Series(train_bad, index=train.index), train_app)
            if train_pair[0] and train_pair[2] <= 0.02:
                rows.append(
                    {
                        "dataset": name,
                        "lhs_column": lhs,
                        "lhs_value": "*",
                        "rhs_column": rhs,
                        "predicted_value": "train-only modal mapping",
                        "support": train_pair[0],
                        "train_confidence": 1.0 - train_pair[2],
                        "test_applicable": int(test_app.sum()),
                        "test_violation_rate": violation_summary(pd.Series(test_bad, index=test.index), test_app)[2],
                        "candidate_kind": "functional_dependency" if train_pair[2] == 0 else "near_functional_dependency",
                    }
                )
    return rows


def profile_dataset(repo: Path, name: str) -> tuple[dict, list[dict], list[dict]]:
    info_path = repo / "data" / name / "info.json"
    real_path = repo / "synthetic" / name / "real.csv"
    test_path = repo / "synthetic" / name / "test.csv"
    if not (info_path.exists() and real_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"{name}: run download_dataset.py and process_dataset.py first")
    info = json.loads(info_path.read_text())
    train = pd.read_csv(real_path)
    test = pd.read_csv(test_path)
    columns = list(train.columns)
    num_idx = list(info.get("num_col_idx", []))
    cat_idx = list(info.get("cat_col_idx", [])) + list(info.get("target_col_idx", []))
    if info.get("task_type") == "regression":
        num_idx += list(info.get("target_col_idx", []))
        cat_idx = list(info.get("cat_col_idx", []))
    num_cols = [columns[i] for i in num_idx]
    cat_cols = [columns[i] for i in cat_idx]

    col_rows = []
    for col in columns:
        is_num = col in num_cols
        vc = train[col].value_counts(dropna=False)
        rare_threshold = max(5, int(0.001 * len(train)))
        col_rows.append(
            {
                "dataset": name,
                "column": col,
                "type": "numeric" if is_num else "categorical",
                "train_missing_rate": float(train[col].isna().mean()),
                "test_missing_rate": float(test[col].isna().mean()),
                "cardinality": int(train[col].nunique(dropna=False)),
                "rare_value_count": int((vc <= rare_threshold).sum()),
                "rare_row_fraction": float(vc[vc <= rare_threshold].sum() / len(train)),
            }
        )
    profile = {
        "dataset": name,
        "train_rows": len(train),
        "test_rows": len(test),
        "columns": len(columns),
        "numeric_columns": len(num_cols),
        "categorical_columns": len(cat_cols),
        "has_long_text": bool(any(train[c].astype(str).str.len().quantile(0.95) > 80 for c in cat_cols)),
        "column_details": col_rows,
    }
    manual = audit_manual_rules(name, train, test)
    discovered = discover_train_only_rules(name, train, test, cat_cols, num_cols)
    return profile, manual, discovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["adult", "default", "shoppers"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    profiles, rules, candidates = [], [], []
    errors = []
    for name in args.datasets:
        try:
            profile, manual, discovered = profile_dataset(args.repo, name)
            profiles.append(profile)
            rules.extend(manual)
            candidates.extend(discovered)
        except Exception as exc:  # keep profiling the other public datasets
            errors.append({"dataset": name, "error": f"{type(exc).__name__}: {exc}"})

    (args.output / "dataset_profiles.json").write_text(json.dumps(profiles, indent=2, default=jsonable))
    pd.DataFrame([row for p in profiles for row in p["column_details"]]).to_csv(args.output / "column_profile.csv", index=False)
    pd.DataFrame(rules).to_csv(args.output / "rule_audit.csv", index=False)
    pd.DataFrame(candidates).to_csv(args.output / "train_only_rule_candidates.csv", index=False)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "errors": errors,
    }
    (args.output / "profile_environment.json").write_text(json.dumps(environment, indent=2))
    print(json.dumps({"datasets_profiled": [p["dataset"] for p in profiles], "rules": len(rules), "candidates": len(candidates), "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
