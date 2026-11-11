#!/usr/bin/env python3
"""Common extension evaluator with classification and regression utility."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def canonicalize(frame: pd.DataFrame, info: dict) -> pd.DataFrame:
    result = frame.copy()
    positions = list(info["cat_col_idx"])
    if info["task_type"] != "regression":
        positions += list(info["target_col_idx"])
    for position in positions:
        column = info["column_names"][position]
        values = result[column].astype(str).str.strip()
        numeric = pd.to_numeric(values, errors="coerce")
        mask = numeric.notna()
        if mask.any():
            values.loc[mask] = numeric.loc[mask].map(lambda value: f"{value:.15g}")
        result[column] = values
    return result


def regression_utility(synthetic: pd.DataFrame, heldout: pd.DataFrame,
                       info: dict, seed: int) -> tuple[float, float]:
    target = info["column_names"][info["target_col_idx"][0]]
    numeric = [info["column_names"][index] for index in info["num_col_idx"]]
    categorical = [info["column_names"][index] for index in info["cat_col_idx"]]
    transform = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = XGBRegressor(
        n_estimators=200, max_depth=5, min_child_weight=1, objective="reg:squarederror",
        tree_method="hist", n_jobs=8, random_state=seed,
    )
    pipeline = make_pipeline(transform, model)
    y_train = pd.to_numeric(synthetic[target], errors="raise").to_numpy(float)
    y_test = pd.to_numeric(heldout[target], errors="raise").to_numpy(float)
    pipeline.fit(synthetic[numeric + categorical], y_train)
    prediction = pipeline.predict(heldout[numeric + categorical])
    rmse = math.sqrt(mean_squared_error(y_test, prediction))
    scale = float(np.std(y_test, ddof=0))
    return float(r2_score(y_test, prediction)), float(rmse / scale) if scale > 0 else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample", action="append", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    from dygraft import evaluation as base
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    real = canonicalize(pd.read_csv(args.repo / "synthetic" / args.dataset / "real.csv"), info)
    heldout = canonicalize(pd.read_csv(args.repo / "synthetic" / args.dataset / "test.csv"), info)
    edges = json.loads(args.graph.read_text())
    rows = []
    for value in args.sample:
        name, path = value.split("=", 1)
        synthetic = canonicalize(pd.read_csv(path), info)[real.columns]
        if args.max_rows > 0 and len(synthetic) > args.max_rows:
            synthetic = synthetic.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        shape, trend, detection = base.quality_and_detection(real, synthetic, info)
        alpha, beta = base.alpha_beta(real, synthetic, info)
        row = {
            "method": name,
            "rows": len(synthetic),
            "shape_error_percent": shape,
            "trend_error_percent": trend,
            "alpha_precision_percent": alpha,
            "beta_recall_percent": beta,
            "detection_score": detection,
            "dcr_percent_50_ideal": base.dcr_score(real, heldout, synthetic, info),
            **base.constraint_metrics(synthetic, heldout, edges),
        }
        if info["task_type"] == "regression":
            r2, nrmse = regression_utility(synthetic, heldout, info, args.seed)
            row["mle_xgboost_r2"] = r2
            row["mle_xgboost_nrmse"] = nrmse
        else:
            row["mle_xgboost_auc"] = base.mle_auc(synthetic, heldout, info, args.seed)
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(args.output / "baseline_metrics.csv", index=False)
    decision = {
        "dataset": args.dataset,
        "task_type": info["task_type"],
        "selection_uses_heldout": False,
        "evaluation_uses_heldout": True,
        "utility": "R2 and RMSE/std(test target)" if info["task_type"] == "regression" else "ROC-AUC",
        "methods": list(results["method"]),
    }
    (args.output / "evaluation_metadata.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(results.to_csv(index=False))


if __name__ == "__main__":
    main()
