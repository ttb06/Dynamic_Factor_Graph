#!/usr/bin/env python3
"""Controlled audit of ambiguity in repairing count-duration implications.

Every observed violation has the form count=0 and duration>0, but it is made
through one of two hidden routes: corrupt a zero-row duration, or corrupt a
positive-row count.  Both deterministic repairs satisfy the rule; only one
restores the clean record.  This isolates route selection from rule parsing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PAIRS = [
    ("Administrative", "Administrative_Duration"),
    ("Informational", "Informational_Duration"),
    ("ProductRelated", "ProductRelated_Duration"),
]


def make_cases(frame: pd.DataFrame, count: str, duration: str, rng: np.random.Generator,
               max_per_route: int) -> pd.DataFrame:
    zero = frame[(frame[count] == 0) & (frame[duration] == 0)].copy()
    positive = frame[(frame[count] > 0) & (frame[duration] > 0)].copy()
    n = min(len(zero), len(positive), max_per_route)
    if n < 20:
        raise RuntimeError(f"insufficient cases for {count}: {n}")
    zero = zero.iloc[rng.choice(len(zero), n, replace=False)].reset_index(drop=True)
    positive = positive.iloc[rng.choice(len(positive), n, replace=False)].reset_index(drop=True)

    # Route 0: duration was corrupted.  Values are donated by positive rows so
    # magnitude alone cannot identify a synthetically out-of-range value.
    donors = positive[duration].to_numpy()[rng.permutation(n)]
    observed_zero = zero.copy()
    observed_zero[duration] = donors
    observed_zero["_route"] = 0
    observed_zero["_clean_count"] = zero[count].to_numpy()
    observed_zero["_clean_duration"] = zero[duration].to_numpy()

    # Route 1: count was corrupted.  The positive duration remains untouched.
    observed_positive = positive.copy()
    observed_positive[count] = 0
    observed_positive["_route"] = 1
    observed_positive["_clean_count"] = positive[count].to_numpy()
    observed_positive["_clean_duration"] = positive[duration].to_numpy()
    cases = pd.concat([observed_zero, observed_positive], ignore_index=True)
    return cases.iloc[rng.permutation(len(cases))].reset_index(drop=True)


def model_features(frame: pd.DataFrame, count: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    hidden = {"Revenue", count, "_route", "_clean_count", "_clean_duration"}
    columns = [c for c in frame.columns if c not in hidden]
    x = frame[columns].copy()
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical = [c for c in columns if c not in numeric]
    return x, numeric, categorical


def fit_router(train_cases: pd.DataFrame, count: str) -> Pipeline:
    x, numeric, categorical = model_features(train_cases, count)
    transform = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("features", transform),
                      ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    model.fit(x, train_cases["_route"])
    return model


def normalized_pair_error(pred_count: np.ndarray, pred_duration: np.ndarray,
                          cases: pd.DataFrame, count_scale: float, duration_scale: float) -> float:
    return float(np.mean(
        np.abs(pred_count - cases["_clean_count"].to_numpy()) / count_scale
        + np.abs(pred_duration - cases["_clean_duration"].to_numpy()) / duration_scale
    ))


def evaluate_routes(cases: pd.DataFrame, predicted_route: np.ndarray, count: str, duration: str,
                    positive_train: pd.DataFrame, train: pd.DataFrame, method: str) -> dict:
    # Route 0 repairs the consequent to zero.  Route 1 restores an antecedent
    # value by median count among duration-nearest positive training records.
    observed_count = cases[count].to_numpy(float)
    observed_duration = cases[duration].to_numpy(float)
    donor_duration = positive_train[duration].to_numpy(float)
    donor_count = positive_train[count].to_numpy(float)
    order = np.argsort(donor_duration)
    donor_duration, donor_count = donor_duration[order], donor_count[order]
    positions = np.searchsorted(donor_duration, observed_duration).clip(0, len(order) - 1)
    left = np.maximum(positions - 1, 0)
    choose_left = np.abs(donor_duration[left] - observed_duration) < np.abs(
        donor_duration[positions] - observed_duration)
    nearest = np.where(choose_left, left, positions)
    imputed_count = np.maximum(donor_count[nearest], 1)

    pred_count = np.where(predicted_route == 1, imputed_count, observed_count)
    pred_duration = np.where(predicted_route == 0, 0.0, observed_duration)
    count_scale = max(float(train[count].std()), 1e-8)
    duration_scale = max(float(train[duration].std()), 1e-8)
    validity = np.mean(~((pred_count == 0) & (pred_duration > 0)))
    truth = cases["_route"].to_numpy(int)
    return {
        "relation": f"{count}=0 -> {duration}=0",
        "method": method,
        "cases": len(cases),
        "route_accuracy": float(accuracy_score(truth, predicted_route)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted_route)),
        "route_f1": float(f1_score(truth, predicted_route)),
        "validity": float(validity),
        "normalized_pair_error": normalized_pair_error(
            pred_count, pred_duration, cases, count_scale, duration_scale),
        "wrong_column_edit_rate": float(np.mean(predicted_route != truth)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-train-per-route", type=int, default=2500)
    parser.add_argument("--max-test-per-route", type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(args.repo / "data/shoppers/train.csv")
    test = pd.read_csv(args.repo / "data/shoppers/test.csv")
    rng = np.random.default_rng(args.seed)
    rows = []
    case_summary = []
    skipped = []
    for count, duration in PAIRS:
        try:
            train_cases = make_cases(train, count, duration, rng, args.max_train_per_route)
            test_cases = make_cases(test, count, duration, rng, args.max_test_per_route)
        except RuntimeError as error:
            skipped.append({"relation": f"{count}=0 -> {duration}=0", "reason": str(error)})
            continue
        router = fit_router(train_cases, count)
        x_test, _, _ = model_features(test_cases, count)
        learned = router.predict(x_test).astype(int)
        truth = test_cases["_route"].to_numpy(int)
        observed_duration = test_cases[duration].to_numpy(float)
        count_scale = max(float(train[count].std()), 1e-8)
        duration_scale = max(float(train[duration].std()), 1e-8)
        # Minimal normalized edit has access only to edit magnitude, not the
        # hidden corruption route.  Count repair is conservatively one unit.
        minimal = (1.0 / count_scale < observed_duration / duration_scale).astype(int)
        positive_train = train[(train[count] > 0) & (train[duration] > 0)]
        methods = {
            "consequent_only": np.zeros(len(test_cases), dtype=int),
            "antecedent_only": np.ones(len(test_cases), dtype=int),
            "minimal_normalized_edit": minimal,
            "context_router": learned,
            "oracle_route": truth,
        }
        for name, prediction in methods.items():
            rows.append(evaluate_routes(test_cases, prediction, count, duration,
                                        positive_train, train, name))
        case_summary.append({
            "relation": f"{count}=0 -> {duration}=0",
            "train_cases": len(train_cases),
            "test_cases": len(test_cases),
            "test_route_0": int((truth == 0).sum()),
            "test_route_1": int((truth == 1).sum()),
        })

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("no relation has enough support for the audit")
    metrics.to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(case_summary).to_csv(args.output / "case_summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.output / "skipped_relations.csv", index=False)
    aggregate = metrics.groupby("method").agg(
        relations=("relation", "nunique"),
        route_accuracy=("route_accuracy", "mean"),
        validity=("validity", "mean"),
        normalized_pair_error=("normalized_pair_error", "mean"),
        wrong_column_edit_rate=("wrong_column_edit_rate", "mean"),
    ).reset_index().sort_values("normalized_pair_error")
    aggregate.to_csv(args.output / "aggregate.csv", index=False)
    lines = [
        "# Ambiguous-repair controlled audit",
        "",
        "Mọi method đều nhận cùng violation `count=0, duration>0`. Ground truth cho biết cột nào đã bị corrupt; ground truth này chỉ được dùng để đánh giá và cho oracle.",
        "",
        "| Method | Relations | Route accuracy | Validity | Pair error | Wrong-column edit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(f"| {row.method} | {row.relations} | {row.route_accuracy:.4f} | {row.validity:.4f} | {row.normalized_pair_error:.4f} | {row.wrong_column_edit_rate:.4f} |")
    lines += [
        "", "## Theo relation", "",
        "| Relation | Method | Route accuracy | Validity | Pair error | Wrong-column edit |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.relation} | {row.method} | {row.route_accuracy:.4f} | {row.validity:.4f} | {row.normalized_pair_error:.4f} | {row.wrong_column_edit_rate:.4f} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    metadata = {
        "status": "executed",
        "seed": args.seed,
        "scope": "controlled corruption audit; no diffusion or LLM claim",
        "relations": int(metrics["relation"].nunique()),
        "skipped_relations": skipped,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
