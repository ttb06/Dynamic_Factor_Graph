#!/usr/bin/env python3
"""Evaluate samples with the TabDiff/TabSyn metric families.

The output deliberately separates paper-comparable general quality metrics
from constraint-catalog metrics introduced for dynamic graph guidance.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import LogisticDetection
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from typed_dynamic_graph_guidance import edge_masks, graph_metrics


class ConvergedLogisticDetection(LogisticDetection):
    """The same SDMetrics detector with a converged, reproducible solver."""

    @staticmethod
    def _get_classifier():
        return LogisticRegression(solver="lbfgs", max_iter=2000,
                                  random_state=20261007)


def reordered(real: pd.DataFrame, synthetic: pd.DataFrame, info: dict):
    num = list(info["num_col_idx"]); cat = list(info["cat_col_idx"])
    (num if info["task_type"] == "regression" else cat).extend(info["target_col_idx"])
    real = real.iloc[:, num + cat].copy(); synthetic = synthetic.iloc[:, num + cat].copy()
    # Current SDMetrics detection treats a missing primary key as an empty
    # string and assumes string column names.  Use string indices to remain
    # compatible with the official metadata and newer SDMetrics releases.
    names = [str(i) for i in range(real.shape[1])]
    real.columns = names; synthetic.columns = names
    original = info["metadata"]["columns"]
    metadata = {"columns": {str(i): copy.deepcopy(original[str(source)])
                            for i, source in enumerate(num + cat)}}
    return real, synthetic, metadata, num, cat


def quality_and_detection(real, synthetic, info):
    r, s, metadata, _, _ = reordered(real, synthetic, info)
    report = QualityReport(); report.generate(r, s, metadata, verbose=False)
    props = report.get_properties().set_index("Property")["Score"]
    # SDMetrics' LogisticDetection creates a shuffled StratifiedKFold without
    # an explicit random_state. Reset NumPy here so identical input tables are
    # guaranteed to receive the same score inside a multi-method evaluation.
    np.random.seed(20261007)
    try:
        detection = ConvergedLogisticDetection.compute(
            real_data=r, synthetic_data=s, metadata=metadata)
    except Exception:
        # Some external baselines collapse a categorical feature to one value.
        # Older SDMetrics then sends that string column through its numeric mean
        # imputer and aborts the whole evaluation.  Preserve the independently
        # computed quality metrics and mark only detection as unavailable.
        detection = float("nan")
    return 100 * (1 - float(props["Column Shapes"])), 100 * (1 - float(props["Column Pair Trends"])), float(detection)


def alpha_beta(real, synthetic, info):
    """Naive AlphaPrecision/BetaRecall used by TabDiff's evaluation script.

    This is the dependency-light implementation of SynthCity's public metric:
    one-hot encode categoricals, min-max scale using real data, then integrate
    the 30-point precision and coverage curves.  Equal-size deterministic
    subsets are used because the metric requires paired sample counts.
    """
    r, s, _, num, cat = reordered(real, synthetic, info)
    n = min(len(r), len(s)); rng = np.random.default_rng(20261007)
    r = r.iloc[rng.choice(len(r), n, replace=False)].reset_index(drop=True)
    s = s.iloc[rng.choice(len(s), n, replace=False)].reset_index(drop=True)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(r.iloc[:, len(num):].astype(str))

    def matrix(frame):
        numeric = frame.iloc[:, :len(num)].to_numpy(float)
        categorical = encoder.transform(frame.iloc[:, len(num):].astype(str))
        return np.concatenate([numeric, categorical], axis=1)

    x = matrix(r); x_syn = matrix(s)
    scaler = MinMaxScaler().fit(x); x = scaler.transform(x); x_syn = scaler.transform(x_syn)
    center = np.mean(x, axis=0); synth_center = np.mean(x_syn, axis=0)
    alphas = np.linspace(0, 1, 30)
    radii = np.quantile(np.linalg.norm(x - center, axis=1), alphas)
    synth_to_center = np.linalg.norm(x_syn - center, axis=1)

    real_nn = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(x)
    real_to_real = real_nn.kneighbors(x, return_distance=True)[0][:, 1]
    synth_nn = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(x_syn)
    real_to_synth, nearest = synth_nn.kneighbors(x, return_distance=True)
    real_to_synth = real_to_synth[:, 0]; nearest = nearest[:, 0]
    closest = x_syn[nearest]
    closest_radius = np.linalg.norm(closest - synth_center, axis=1)
    closest_quantiles = np.quantile(closest_radius, alphas)
    precision_curve = np.asarray([np.mean(synth_to_center <= radius) for radius in radii])
    coverage_curve = np.asarray([
        np.mean((real_to_synth <= real_to_real) & (closest_radius <= radius))
        for radius in closest_quantiles
    ])
    alpha_precision = 1 - np.abs(alphas - precision_curve).sum() / alphas.sum()
    beta_recall = 1 - np.abs(alphas - coverage_curve).sum() / alphas.sum()
    return 100 * float(alpha_precision), 100 * float(beta_recall)


def dcr_score(real, heldout, synthetic, info, batch=256):
    # A nearest-neighbour comparison is biased toward the larger reference
    # pool.  Equalize train and held-out pool sizes so 50% is interpretable as
    # the no-preference point used by the standard DCR privacy diagnostic.
    n = min(len(real), len(heldout))
    rng = np.random.default_rng(20261007)
    real = real.iloc[rng.choice(len(real), n, replace=False)].reset_index(drop=True)
    heldout = heldout.iloc[rng.choice(len(heldout), n, replace=False)].reset_index(drop=True)
    _, _, _, num, cat = reordered(real, synthetic, info)
    ranges = real.iloc[:, num].max().to_numpy(float) - real.iloc[:, num].min().to_numpy(float)
    ranges[ranges == 0] = 1
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(pd.concat([real.iloc[:, cat], heldout.iloc[:, cat], synthetic.iloc[:, cat]]).astype(str))
    def matrix(frame):
        numeric = frame.iloc[:, num].to_numpy(float) / ranges
        categorical = encoder.transform(frame.iloc[:, cat].astype(str))
        return np.concatenate([numeric, categorical], axis=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = torch.as_tensor(matrix(real), device=device); h = torch.as_tensor(matrix(heldout), device=device)
    s = torch.as_tensor(matrix(synthetic), device=device); closer = total = 0
    for start in range(0, len(s), batch):
        sb = s[start:start + batch]; dr = torch.full((len(sb),), float("inf"), device=device)
        dh = torch.full((len(sb),), float("inf"), device=device)
        for other, target in ((r, dr), (h, dh)):
            for offset in range(0, len(other), 1024):
                distance = (sb[:, None] - other[offset:offset + 1024]).abs().sum(dim=2).min(dim=1).values
                if other.data_ptr() == r.data_ptr(): dr = torch.minimum(dr, distance)
                else: dh = torch.minimum(dh, distance)
        closer += int((dr < dh).sum()); total += len(sb)
    return 100 * closer / total


def mle_auc(synthetic, heldout, info, seed):
    target = info["column_names"][info["target_col_idx"][0]]
    numeric = [info["column_names"][i] for i in info["num_col_idx"]]
    categorical = [info["column_names"][i] for i in info["cat_col_idx"]]
    label = LabelEncoder().fit(pd.concat([synthetic[target], heldout[target]]).astype(str))
    y_train = label.transform(synthetic[target].astype(str)); y_test = label.transform(heldout[target].astype(str))
    transform = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = XGBClassifier(n_estimators=100, max_depth=5, min_child_weight=1, gamma=0,
                          objective="binary:logistic", tree_method="hist", n_jobs=8,
                          random_state=seed, eval_metric="logloss")
    pipeline = make_pipeline(transform, model)
    pipeline.fit(synthetic[numeric + categorical], y_train)
    return float(roc_auc_score(y_test, pipeline.predict_proba(heldout[numeric + categorical])[:, 1]))


def constraint_metrics(frame, reference, edges):
    result = graph_metrics(frame, reference, edges)
    for role in ("actuate", "score_only", "monitor"):
        subset = [edge for edge in edges if edge.get("usage_role", "actuate") == role]
        if subset:
            for key, value in graph_metrics(frame, reference, subset).items(): result[f"{role}_{key}"] = value
    # The sampler is authorized to modify actuate edges and to use score-only
    # edges as branch-ranking guardrails.  Monitor edges are deliberately
    # outside its optimization target, so report the actionable subgraph as a
    # separate primary diagnostic rather than diluting it with audit-only
    # relations.  Full-graph metrics remain unchanged and are still reported.
    actionable = [edge for edge in edges if edge.get("usage_role", "actuate") != "monitor"]
    if actionable:
        for key, value in graph_metrics(frame, reference, actionable).items():
            result[f"actionable_{key}"] = value
    supported = 0; active_pairs = 0
    for edge in edges:
        active = edge_masks(frame, edge)[0]; active_pairs += int(active.sum()); supported += int(active.sum() >= 30)
    result["catalog_edge_coverage"] = supported / len(edges)
    result["active_row_edge_fraction"] = active_pairs / (len(frame) * len(edges))
    return result


def parse_sample(value: str):
    name, path = value.split("=", 1); return name, Path(path)


def canonicalize_categories(frame: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Normalize representation-only whitespace on categorical values.

    Adult's raw split retains leading spaces from the original CSV whereas
    decoded generators emit the same labels without those spaces.  Treating
    them as distinct values corrupts both quality metrics and downstream MLE
    labels (for example, a binary target appears to have four classes).
    """
    result = frame.copy()
    positions = list(info["cat_col_idx"]) + list(info["target_col_idx"])
    for position in positions:
        column = info["column_names"][position]
        result[column] = result[column].astype(str).str.strip()
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    """Render the small result table without pandas' optional tabulate extra."""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    header = [str(column) for column in display.columns]
    return "\n".join([
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset", default="shoppers")
    parser.add_argument("--sample", action="append", required=True, help="METHOD=/path/to/sample.csv")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Deterministically subsample every method to the same row budget; 0 keeps all rows.")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo))
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    real = canonicalize_categories(
        pd.read_csv(args.repo / "synthetic" / args.dataset / "real.csv"), info)
    heldout = canonicalize_categories(
        pd.read_csv(args.repo / "synthetic" / args.dataset / "test.csv"), info)
    edges = json.loads(args.graph.read_text()); rows = []; per_edge = []
    for name, path in map(parse_sample, args.sample):
        synthetic = canonicalize_categories(pd.read_csv(path), info)
        synthetic = synthetic[real.columns]
        if args.max_rows > 0 and len(synthetic) > args.max_rows:
            synthetic = synthetic.sample(
                n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        shape, trend, detection = quality_and_detection(real, synthetic, info)
        alpha, beta = alpha_beta(real, synthetic, info)
        row = {"method": name, "rows": len(synthetic), "shape_error_percent": shape,
               "trend_error_percent": trend, "alpha_precision_percent": alpha,
               "beta_recall_percent": beta, "detection_score": detection,
               "dcr_percent_50_ideal": dcr_score(real, heldout, synthetic, info),
               "mle_xgboost_auc": mle_auc(synthetic, heldout, info, args.seed),
               **constraint_metrics(synthetic, heldout, edges)}
        rows.append(row)
        for edge in edges:
            active, _, bad = edge_masks(synthetic, edge)
            per_edge.append({"method": name, "candidate_id": edge["candidate_id"],
                             "usage_role": edge.get("usage_role", "actuate"),
                             "support": int(active.sum()), "violations": int(bad.sum()),
                             "violation_rate": float(bad.sum() / active.sum()) if active.sum() else math.nan})
    results = pd.DataFrame(rows); results.to_csv(args.output / "baseline_metrics.csv", index=False)
    pd.DataFrame(per_edge).to_csv(args.output / "baseline_constraint_metrics_by_edge.csv", index=False)
    columns = ["method", "shape_error_percent", "trend_error_percent", "alpha_precision_percent",
               "beta_recall_percent", "detection_score", "mle_xgboost_auc", "dcr_percent_50_ideal",
               "actuate_macro_violation", "score_only_macro_violation", "actionable_macro_violation",
               "actionable_heldout_violation_error", "heldout_violation_error"]
    # Graphs used by focused mechanism studies may contain no score-only edges.
    # Keep the common evaluator usable for those graphs instead of failing only
    # while rendering a table after all expensive metrics have completed.
    columns = [column for column in columns if column in results.columns]
    lines = [f"# Locally evaluated {args.dataset} baselines", "", markdown_table(results[columns]), "",
             "All rows use the same repository train/test split, sample count, catalog, and metric implementation."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__": main()
