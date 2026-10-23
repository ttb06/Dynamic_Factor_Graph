#!/usr/bin/env python3
"""Common evaluation for ICLR-2024 News constraints and TabDiff fidelity."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sdmetrics.reports.single_table import QualityReport

from evaluate_shoppers_baseline_suite import alpha_beta, quality_and_detection, reordered


RELATIONS = [
    ("tokens_title_le_content", " n_tokens_title", " n_tokens_content"),
    ("nonstop_unique_le_nonstop_words", " n_non_stop_unique_tokens", " n_non_stop_words"),
    ("self_hrefs_le_hrefs", " num_self_hrefs", " num_hrefs"),
    ("selfref_min_le_max", " self_reference_min_shares", " self_reference_max_shares"),
    ("selfref_avg_le_max", " self_reference_avg_sharess", " self_reference_max_shares"),
]


def from_cdgm_pickle(path: Path, partition: str, round_index: int) -> pd.DataFrame:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    return pd.DataFrame(obj[partition][round_index]).copy()


def normalize(frame: pd.DataFrame, reference_columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if "data_channel" not in frame:
        candidates = [column for column in frame if column.startswith("data_channel_is_")]
        if candidates:
            values = frame[candidates].to_numpy(float).argmax(axis=1)
            frame["data_channel"] = [candidates[index].replace("data_channel_is_", "") for index in values]
    if "weekday" not in frame:
        candidates = [column for column in frame if column.startswith("weekday_is_")]
        if candidates:
            values = frame[candidates].to_numpy(float).argmax(axis=1)
            frame["weekday"] = [candidates[index].replace("weekday_is_", "") for index in values]
    stripped = {column.strip(): column for column in reference_columns}
    missing = [name for name in stripped if name not in frame]
    if missing:
        raise ValueError(f"Missing normalized News columns: {missing}")
    result = frame[list(stripped)].copy()
    result.columns = [stripped[name] for name in result]
    return result[reference_columns]


def regression_utility(synthetic: pd.DataFrame, heldout: pd.DataFrame, info: dict, seed: int) -> tuple[float, float]:
    target = info["column_names"][info["target_col_idx"][0]]
    numeric = [info["column_names"][i] for i in info["num_col_idx"]]
    categorical = [info["column_names"][i] for i in info["cat_col_idx"]]
    transform = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                         tree_method="hist", n_jobs=8, random_state=seed)
    pipeline = make_pipeline(transform, model)
    train_x = synthetic[numeric + categorical].copy()
    test_x = heldout[numeric + categorical].copy()
    for column in numeric:
        train_x[column] = pd.to_numeric(train_x[column], errors="coerce")
        test_x[column] = pd.to_numeric(test_x[column], errors="coerce")
    for column in categorical:
        # External generators may collapse a categorical field or encode
        # missing values with a float NaN.  Normalize both sides to the same
        # string domain so OneHotEncoder sees a stable dtype.
        train_x[column] = train_x[column].fillna("<MISSING>").astype(str)
        test_x[column] = test_x[column].fillna("<MISSING>").astype(str)
    pipeline.fit(train_x, pd.to_numeric(synthetic[target]))
    truth = pd.to_numeric(heldout[target]).to_numpy(float)
    prediction = pipeline.predict(test_x)
    nrmse = mean_squared_error(truth, prediction) ** 0.5 / max(float(np.std(truth)), 1e-12)
    return float(r2_score(truth, prediction)), float(nrmse)


def quality_without_detection(real: pd.DataFrame, synthetic: pd.DataFrame,
                              info: dict) -> tuple[float, float, float]:
    """Compute the same Shape/Trend scores without the slow detector."""
    r, s, metadata, _, _ = reordered(real, synthetic, info)
    report = QualityReport()
    report.generate(r, s, metadata, verbose=False)
    properties = report.get_properties().set_index("Property")["Score"]
    return (
        100.0 * (1.0 - float(properties["Column Shapes"])),
        100.0 * (1.0 - float(properties["Column Pair Trends"])),
        float("nan"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sample", action="append", default=[], help="METHOD=csv")
    parser.add_argument("--cdgm-pickle", action="append", default=[], help="METHOD=generated_data.pkl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--skip-detection", action="store_true",
                        help="Keep Shape/Trend but skip the slow logistic detector during screens.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.repo / "data/news/info.json").read_text())
    train = pd.read_csv(args.repo / "synthetic/news/real.csv")
    heldout = pd.read_csv(args.repo / "synthetic/news/test.csv")
    entries: list[tuple[str, pd.DataFrame]] = []
    for item in args.sample:
        name, path = item.split("=", 1)
        entries.append((name, pd.read_csv(path)))
    for item in args.cdgm_pickle:
        name, path = item.split("=", 1)
        entries.append((name, from_cdgm_pickle(Path(path), "train", 0)))
    rows = []
    for name, raw in entries:
        sample = normalize(raw, list(train.columns))
        if args.skip_detection:
            shape, trend, detection = quality_without_detection(train, sample, info)
        else:
            shape, trend, detection = quality_and_detection(train, sample, info)
        alpha, beta = alpha_beta(train, sample, info)
        utility_r2, utility_nrmse = regression_utility(sample, heldout, info, args.seed)
        violations = {}
        all_valid = np.ones(len(sample), dtype=bool)
        for label, left, right in RELATIONS:
            valid = pd.to_numeric(sample[left], errors="coerce").le(pd.to_numeric(sample[right], errors="coerce") + 1e-2).to_numpy()
            violations[f"violation_{label}"] = float(1.0 - valid.mean())
            all_valid &= valid
        rows.append({
            "method": name, "rows": len(sample),
            "shape_score_percent": 100.0 - shape,
            "trend_score_percent": 100.0 - trend,
            "alpha_precision_percent": alpha, "beta_recall_percent": beta,
            "detection_score": detection,
            "xgboost_regression_r2": utility_r2,
            "xgboost_regression_nrmse": utility_nrmse,
            "any_constraint_violation_percent": 100.0 * (1.0 - all_valid.mean()),
            **violations,
        })
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "metrics.csv", index=False)
    columns = ["method", "shape_score_percent", "trend_score_percent", "alpha_precision_percent",
               "beta_recall_percent", "detection_score", "xgboost_regression_r2",
               "xgboost_regression_nrmse", "any_constraint_violation_percent"]
    lines = ["# Public-constraint News comparison", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in result[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row) + " |")
    lines += ["", "Constraints are the five public News inequalities from ConstrainedDGM (ICLR 2024).",
              "The text constraint file is authoritative; its second inequality differs from the repository's standalone NumPy audit helper.",
              "Fidelity and utility are recomputed locally on the same TabDiff News train/test data."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
