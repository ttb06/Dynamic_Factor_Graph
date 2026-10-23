#!/usr/bin/env python3
"""Train-only balanced particle resampling for an oversampled tabular pool.

The sampler never edits a generated row and never reads validation/test data.
It estimates optional train/pool density ratios, rakes them toward train-only
univariate and pairwise moments, draws several without-replacement particle
sets, and keeps the set with the smallest predeclared moment discrepancy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class MomentGroup:
    name: str
    train_codes: np.ndarray
    pool_codes: np.ndarray
    cardinality: int
    family: str


@dataclass(frozen=True)
class GeometryCoverage:
    nearest_indices: np.ndarray
    nearest_distances: np.ndarray
    real_radii: np.ndarray
    reference_score: float

    def score(self, indices: np.ndarray, pool_rows: int) -> float:
        selected = np.zeros(pool_rows, dtype=bool)
        selected[indices] = True
        covered = selected[self.nearest_indices] & (
            self.nearest_distances <= self.real_radii[:, None])
        return float(np.mean(np.any(covered, axis=1)))


@dataclass(frozen=True)
class IntegratedCoverage:
    train: pd.DataFrame
    pool: pd.DataFrame
    info: dict
    fold_seeds: tuple[int, ...]
    reference_score: float

    def score(self, indices: np.ndarray) -> tuple[float, float]:
        synthetic = self.pool.iloc[indices].reset_index(drop=True)
        values = [
            integrated_alpha_beta(self.train, synthetic, self.info, seed)
            for seed in self.fold_seeds
        ]
        alpha = float(np.mean([value[0] for value in values]))
        beta = float(min(value[1] for value in values))
        return alpha, beta


def typed_columns(info: dict) -> tuple[list[str], list[str]]:
    names = info["column_names"]
    numeric_indices = list(info["num_col_idx"])
    categorical_indices = list(info["cat_col_idx"])
    if info.get("task_type") == "regression":
        numeric_indices.extend(info["target_col_idx"])
    else:
        categorical_indices.extend(info["target_col_idx"])
    numeric = list(dict.fromkeys(names[index] for index in numeric_indices))
    categorical = list(dict.fromkeys(names[index] for index in categorical_indices))
    return numeric, categorical


def canonicalize(frame: pd.DataFrame, info: dict) -> pd.DataFrame:
    result = frame.copy()
    names = info["column_names"]
    numeric, categorical = typed_columns(info)
    for column in categorical:
        result[column] = result[column].astype(str).str.strip()
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result[names]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def categorical_codes(train: pd.Series, pool: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    values = sorted(set(train.astype(str)))
    mapping = {value: index for index, value in enumerate(values)}
    unseen = len(mapping)
    train_codes = train.astype(str).map(mapping).fillna(unseen).to_numpy(np.int64)
    pool_codes = pool.astype(str).map(mapping).fillna(unseen).to_numpy(np.int64)
    return train_codes, pool_codes, unseen + 1


def numerical_codes(train: pd.Series, pool: pd.Series, bins: int) -> tuple[np.ndarray, np.ndarray, int]:
    values = train.to_numpy(float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.zeros(len(train), dtype=np.int64), np.zeros(len(pool), dtype=np.int64), 1
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return np.zeros(len(train), dtype=np.int64), np.zeros(len(pool), dtype=np.int64), 1
    internal = edges[1:-1]
    train_codes = np.searchsorted(internal, train.to_numpy(float), side="right").astype(np.int64)
    pool_codes = np.searchsorted(internal, pool.to_numpy(float), side="right").astype(np.int64)
    return train_codes, pool_codes, len(internal) + 1


def combine_codes(left: np.ndarray, right: np.ndarray, right_cardinality: int) -> np.ndarray:
    return left.astype(np.int64) * int(right_cardinality) + right.astype(np.int64)


def build_groups(train: pd.DataFrame, pool: pd.DataFrame, info: dict,
                 bins: int, pair_bins: int, max_pairs: int,
                 include_pairs: bool, target_pair_quota: int = 0) -> list[MomentGroup]:
    names = info["column_names"]
    numeric, categorical = typed_columns(info)
    univariate: dict[str, MomentGroup] = {}
    pair_basis: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    groups: list[MomentGroup] = []
    for column in numeric:
        tr, po, cardinality = numerical_codes(train[column], pool[column], bins)
        group = MomentGroup(f"uni::{column}", tr, po, cardinality, "univariate")
        univariate[column] = group
        groups.append(group)
        pair_basis[column] = numerical_codes(train[column], pool[column], pair_bins)
    for column in categorical:
        tr, po, cardinality = categorical_codes(train[column], pool[column])
        group = MomentGroup(f"uni::{column}", tr, po, cardinality, "univariate")
        univariate[column] = group
        groups.append(group)
        pair_basis[column] = (tr, po, cardinality)
    if not include_pairs or max_pairs <= 0:
        return groups

    candidates: list[tuple[float, str, str]] = []
    for left_index, left in enumerate(categorical):
        for right in categorical[left_index + 1:]:
            score = float(mutual_info_score(train[left].astype(str), train[right].astype(str)))
            candidates.append((score, left, right))
    numeric_frame = train[numeric].rank(pct=True).corr(method="pearson").abs()
    for left_index, left in enumerate(numeric):
        for right in numeric[left_index + 1:]:
            candidates.append((float(numeric_frame.loc[left, right]), left, right))
    target = names[info["target_col_idx"][0]]
    if target in categorical:
        target_codes = pair_basis[target][0]
        for column in numeric:
            codes = pair_basis[column][0]
            score = float(mutual_info_score(codes, target_codes))
            candidates.append((score, column, target))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if target_pair_quota > 0:
        target_first = [item for item in candidates if target in item[1:]]
        remaining = [item for item in candidates if target not in item[1:]]
        candidates = target_first[:target_pair_quota] + remaining + target_first[target_pair_quota:]
    chosen: set[tuple[str, str]] = set()
    for score, left, right in candidates:
        key = tuple(sorted((left, right)))
        if key in chosen:
            continue
        chosen.add(key)
        left_train, left_pool, _ = pair_basis[left]
        right_train, right_pool, right_cardinality = pair_basis[right]
        train_codes = combine_codes(left_train, right_train, right_cardinality)
        pool_codes = combine_codes(left_pool, right_pool, right_cardinality)
        cardinality = int(max(train_codes.max(initial=0), pool_codes.max(initial=0)) + 1)
        groups.append(MomentGroup(
            f"pair::{left}::{right}", train_codes, pool_codes,
            cardinality, "pair"))
        if len(chosen) >= max_pairs:
            break
    return groups


def target_distribution(group: MomentGroup, smoothing: float = 0.5) -> np.ndarray:
    counts = np.bincount(group.train_codes, minlength=group.cardinality).astype(float)
    counts += smoothing
    return counts / counts.sum()


def build_geometry_coverage(train: pd.DataFrame, pool: pd.DataFrame, info: dict,
                            count: int, neighbors: int) -> GeometryCoverage:
    numeric, categorical = typed_columns(info)
    train_numeric = train[numeric].to_numpy(float)
    pool_numeric = pool[numeric].to_numpy(float)
    low = np.nanmin(train_numeric, axis=0)
    span = np.nanmax(train_numeric, axis=0) - low
    span[span == 0] = 1.0
    train_numeric = (train_numeric - low) / span
    pool_numeric = (pool_numeric - low) / span
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_categorical = encoder.fit_transform(train[categorical].astype(str))
        pool_categorical = encoder.transform(pool[categorical].astype(str))
        train_matrix = np.concatenate([train_numeric, train_categorical], axis=1)
        pool_matrix = np.concatenate([pool_numeric, pool_categorical], axis=1)
    else:
        train_matrix, pool_matrix = train_numeric, pool_numeric
    train_matrix = np.asarray(train_matrix, dtype=np.float32)
    pool_matrix = np.asarray(pool_matrix, dtype=np.float32)
    real_nn = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(train_matrix)
    real_radii = real_nn.kneighbors(train_matrix, return_distance=True)[0][:, 1]
    pool_nn = NearestNeighbors(
        n_neighbors=min(neighbors, len(pool)), n_jobs=-1, p=2).fit(pool_matrix)
    nearest_distances, nearest_indices = pool_nn.kneighbors(
        train_matrix, return_distance=True)
    proxy = GeometryCoverage(
        nearest_indices=np.asarray(nearest_indices, dtype=np.int64),
        nearest_distances=np.asarray(nearest_distances, dtype=np.float32),
        real_radii=np.asarray(real_radii, dtype=np.float32),
        reference_score=0.0)
    reference = proxy.score(np.arange(count, dtype=np.int64), len(pool))
    return GeometryCoverage(
        nearest_indices=proxy.nearest_indices,
        nearest_distances=proxy.nearest_distances,
        real_radii=proxy.real_radii,
        reference_score=reference)


def integrated_alpha_beta(real: pd.DataFrame, synthetic: pd.DataFrame,
                          info: dict, seed: int) -> tuple[float, float]:
    numeric, categorical = typed_columns(info)
    n = min(len(real), len(synthetic))
    rng = np.random.default_rng(seed)
    real = real.iloc[rng.choice(len(real), n, replace=False)].reset_index(drop=True)
    synthetic = synthetic.iloc[rng.choice(len(synthetic), n, replace=False)].reset_index(drop=True)
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(real[categorical].astype(str))
        real_cat = encoder.transform(real[categorical].astype(str))
        synthetic_cat = encoder.transform(synthetic[categorical].astype(str))
    else:
        real_cat = np.empty((n, 0), dtype=float)
        synthetic_cat = np.empty((n, 0), dtype=float)
    real_num = real[numeric].to_numpy(float)
    synthetic_num = synthetic[numeric].to_numpy(float)
    real_matrix = np.concatenate([real_num, real_cat], axis=1)
    synthetic_matrix = np.concatenate([synthetic_num, synthetic_cat], axis=1)
    low = np.nanmin(real_matrix, axis=0)
    span = np.nanmax(real_matrix, axis=0) - low
    span[span == 0] = 1.0
    real_matrix = (real_matrix - low) / span
    synthetic_matrix = (synthetic_matrix - low) / span
    center = np.mean(real_matrix, axis=0)
    synthetic_center = np.mean(synthetic_matrix, axis=0)
    alphas = np.linspace(0, 1, 30)
    radii = np.quantile(np.linalg.norm(real_matrix - center, axis=1), alphas)
    synthetic_to_center = np.linalg.norm(synthetic_matrix - center, axis=1)
    real_nn = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(real_matrix)
    real_to_real = real_nn.kneighbors(real_matrix, return_distance=True)[0][:, 1]
    synthetic_nn = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(synthetic_matrix)
    real_to_synthetic, nearest = synthetic_nn.kneighbors(real_matrix, return_distance=True)
    real_to_synthetic = real_to_synthetic[:, 0]
    closest = synthetic_matrix[nearest[:, 0]]
    closest_radius = np.linalg.norm(closest - synthetic_center, axis=1)
    closest_quantiles = np.quantile(closest_radius, alphas)
    precision_curve = np.asarray([
        np.mean(synthetic_to_center <= radius) for radius in radii])
    coverage_curve = np.asarray([
        np.mean((real_to_synthetic <= real_to_real) & (closest_radius <= radius))
        for radius in closest_quantiles])
    alpha_precision = 1.0 - np.abs(alphas - precision_curve).sum() / alphas.sum()
    beta_recall = 1.0 - np.abs(alphas - coverage_curve).sum() / alphas.sum()
    return float(alpha_precision), float(beta_recall)


def build_integrated_coverage(train: pd.DataFrame, pool: pd.DataFrame,
                              info: dict, count: int, seed: int,
                              folds: int) -> IntegratedCoverage:
    fold_seeds = tuple(seed + 7919 * (index + 1) for index in range(folds))
    provisional = IntegratedCoverage(train, pool, info, fold_seeds, 0.0)
    _, reference = provisional.score(np.arange(count, dtype=np.int64))
    return IntegratedCoverage(train, pool, info, fold_seeds, reference)


def moment_loss(indices: np.ndarray, groups: Sequence[MomentGroup],
                pair_weight: float, coverage_weight: float) -> tuple[float, float, float, float]:
    family_losses: dict[str, list[float]] = {"univariate": [], "pair": []}
    coverage_losses: list[float] = []
    for group in groups:
        target = target_distribution(group)
        observed = np.bincount(group.pool_codes[indices], minlength=group.cardinality).astype(float)
        observed /= max(observed.sum(), 1.0)
        family_losses[group.family].append(float(np.abs(target - observed).sum()))
        deficit = np.maximum(target - observed, 0.0)
        coverage_losses.append(float(np.mean(
            deficit / np.sqrt(np.clip(target, 1e-8, None)))))
    univariate = float(np.mean(family_losses["univariate"])) if family_losses["univariate"] else 0.0
    pair = float(np.mean(family_losses["pair"])) if family_losses["pair"] else 0.0
    coverage = float(np.mean(coverage_losses)) if coverage_losses else 0.0
    return univariate + pair_weight * pair + coverage_weight * coverage, univariate, pair, coverage


def density_ratio(train: pd.DataFrame, pool: pd.DataFrame, info: dict,
                  model: str, seed: int, clip: float) -> tuple[np.ndarray, dict]:
    numeric, categorical = typed_columns(info)
    rng = np.random.default_rng(seed)
    take = min(len(train), len(pool))
    train_fit = train.iloc[rng.choice(len(train), take, replace=False)].copy()
    pool_fit_indices = rng.choice(len(pool), take, replace=False)
    pool_fit = pool.iloc[pool_fit_indices].copy()
    x = pd.concat([train_fit, pool_fit], ignore_index=True)
    y = np.concatenate([np.ones(take, dtype=int), np.zeros(take, dtype=int)])
    transform = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    if model == "logit":
        classifier = LogisticRegression(
            solver="lbfgs", C=0.5, max_iter=2000,
            random_state=seed)
    elif model == "xgb":
        from xgboost import XGBClassifier
        classifier = XGBClassifier(
            n_estimators=160, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
            reg_lambda=2.0, objective="binary:logistic", tree_method="hist",
            n_jobs=8, random_state=seed, eval_metric="logloss")
    else:
        raise ValueError(f"unsupported density-ratio model: {model}")
    pipeline = make_pipeline(transform, classifier)
    pipeline.fit(x, y)
    fit_probability = pipeline.predict_proba(x)[:, 1]
    pool_probability = np.clip(pipeline.predict_proba(pool)[:, 1], 1e-4, 1 - 1e-4)
    weights = pool_probability / (1.0 - pool_probability)
    weights = np.clip(weights, 1.0 / clip, clip)
    diagnostics = {
        "classifier": model,
        "fit_auc": float(roc_auc_score(y, fit_probability)),
        "raw_weight_min": float(weights.min()),
        "raw_weight_max": float(weights.max()),
        "raw_weight_mean": float(weights.mean()),
    }
    return weights, diagnostics


def rake(weights: np.ndarray, groups: Sequence[MomentGroup], iterations: int,
         damping: float, clip: float) -> tuple[np.ndarray, list[dict]]:
    result = np.asarray(weights, dtype=float).copy()
    trace: list[dict] = []
    for iteration in range(iterations):
        for group in groups:
            target = target_distribution(group)
            observed = np.bincount(
                group.pool_codes, weights=result,
                minlength=group.cardinality).astype(float)
            observed /= max(observed.sum(), 1e-12)
            factors = np.power(target / np.clip(observed, 1e-8, None), damping)
            result *= factors[group.pool_codes]
            result /= max(result.mean(), 1e-12)
            result = np.clip(result, 1.0 / clip, clip)
        weighted_losses = []
        for group in groups:
            target = target_distribution(group)
            observed = np.bincount(
                group.pool_codes, weights=result,
                minlength=group.cardinality).astype(float)
            observed /= max(observed.sum(), 1e-12)
            weighted_losses.append(float(np.abs(target - observed).sum()))
        trace.append({
            "iteration": iteration + 1,
            "mean_group_l1": float(np.mean(weighted_losses)),
            "weight_min": float(result.min()),
            "weight_max": float(result.max()),
        })
    return result, trace


def sample_best(weights: np.ndarray, groups: Sequence[MomentGroup], count: int,
                draws: int, seed: int, tail_mix: float,
                temperature: float, pair_weight: float,
                coverage_weight: float, geometry: GeometryCoverage | None,
                geometry_weight: float,
                geometry_tolerance: float,
                integrated: IntegratedCoverage | None,
                integrated_weight: float,
                integrated_tolerance: float,
                integrated_shortlist: int) -> tuple[np.ndarray, dict]:
    if not 0 < count < len(weights):
        raise ValueError("count must be positive and smaller than pool size")
    probabilities = np.power(np.clip(weights, 1e-12, None), temperature)
    probabilities /= probabilities.sum()
    probabilities = (1.0 - tail_mix) * probabilities + tail_mix / len(probabilities)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    candidate_indices: list[np.ndarray] = []
    for draw in range(draws):
        indices = np.sort(rng.choice(len(weights), size=count, replace=False, p=probabilities))
        base_objective, univariate, pair, coverage = moment_loss(
            indices, groups, pair_weight, coverage_weight)
        geometry_score = geometry.score(indices, len(weights)) if geometry else 0.0
        geometry_deficit = (
            max(geometry.reference_score - geometry_tolerance - geometry_score, 0.0)
            if geometry else 0.0)
        objective = base_objective + geometry_weight * geometry_deficit
        record = {
            "draw": draw, "objective": objective,
            "base_objective": base_objective,
            "univariate_l1": univariate, "pair_l1": pair,
            "rare_cell_coverage_loss": coverage,
            "geometric_coverage": geometry_score,
            "geometric_coverage_deficit": geometry_deficit,
            "integrated_alpha": None,
            "integrated_coverage": None,
            "integrated_coverage_deficit": None,
        }
        records.append(record)
        candidate_indices.append(indices)
    eligible_draws = list(range(draws))
    if integrated is not None:
        eligible_draws = sorted(
            eligible_draws, key=lambda index: (records[index]["objective"], index)
        )[:integrated_shortlist]
        for index in eligible_draws:
            alpha, beta = integrated.score(candidate_indices[index])
            deficit = max(
                integrated.reference_score - integrated_tolerance - beta, 0.0)
            records[index]["integrated_alpha"] = alpha
            records[index]["integrated_coverage"] = beta
            records[index]["integrated_coverage_deficit"] = deficit
            records[index]["objective"] += integrated_weight * deficit
    best_draw = min(
        eligible_draws, key=lambda index: (records[index]["objective"], index))
    best_indices = candidate_indices[best_draw]
    best_record = records[best_draw]
    uniform_indices = np.sort(rng.choice(len(weights), size=count, replace=False))
    uniform_base, uniform_uni, uniform_pair, uniform_coverage = moment_loss(
        uniform_indices, groups, pair_weight, coverage_weight)
    uniform_geometry = geometry.score(uniform_indices, len(weights)) if geometry else 0.0
    uniform_deficit = (
        max(geometry.reference_score - geometry_tolerance - uniform_geometry, 0.0)
        if geometry else 0.0)
    uniform_objective = uniform_base + geometry_weight * uniform_deficit
    uniform_integrated_alpha = 0.0
    uniform_integrated_beta = 0.0
    uniform_integrated_deficit = 0.0
    if integrated is not None:
        uniform_integrated_alpha, uniform_integrated_beta = integrated.score(uniform_indices)
        uniform_integrated_deficit = max(
            integrated.reference_score - integrated_tolerance - uniform_integrated_beta, 0.0)
        uniform_objective += integrated_weight * uniform_integrated_deficit
    diagnostics = {
        "best": best_record,
        "median_objective": float(np.median([record["objective"] for record in records])),
        "uniform_control": {
            "objective": uniform_objective,
            "base_objective": uniform_base,
            "univariate_l1": uniform_uni,
            "pair_l1": uniform_pair,
            "rare_cell_coverage_loss": uniform_coverage,
            "geometric_coverage": uniform_geometry,
            "geometric_coverage_deficit": uniform_deficit,
            "integrated_alpha": uniform_integrated_alpha,
            "integrated_coverage": uniform_integrated_beta,
            "integrated_coverage_deficit": uniform_integrated_deficit,
        },
        "records": records,
        "uniform_indices": uniform_indices,
        "effective_sample_size": float(1.0 / np.square(probabilities).sum()),
    }
    return best_indices, diagnostics


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--backbone-pool", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--mode", choices=[
        "uniform_search", "rake_marginal", "rake_pair",
        "rake_marginal_pair_select", "ratio_logit", "ratio_xgb",
        "hybrid_logit", "hybrid_xgb", "consensus", "consensus_rake"], required=True)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--pair-bins", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--target-pair-quota", type=int, default=0)
    parser.add_argument("--pair-weight", type=float, default=0.75)
    parser.add_argument("--coverage-weight", type=float, default=0.0)
    parser.add_argument("--prior-index", action="append", type=Path, default=[])
    parser.add_argument("--prior-weight", action="append", type=float, default=[])
    parser.add_argument("--consensus-strength", type=float, default=1.0)
    parser.add_argument("--geometric-coverage-weight", type=float, default=0.0)
    parser.add_argument("--geometric-coverage-k", type=int, default=16)
    parser.add_argument("--geometric-coverage-tolerance", type=float, default=0.0)
    parser.add_argument("--integrated-coverage-weight", type=float, default=0.0)
    parser.add_argument("--integrated-coverage-tolerance", type=float, default=0.0)
    parser.add_argument("--integrated-coverage-shortlist", type=int, default=8)
    parser.add_argument("--integrated-coverage-folds", type=int, default=2)
    parser.add_argument("--rake-iterations", type=int, default=5)
    parser.add_argument("--rake-damping", type=float, default=0.5)
    parser.add_argument("--weight-clip", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--tail-mix", type=float, default=0.10)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261380)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if (not 0.0 <= args.tail_mix <= 1.0 or args.temperature <= 0 or
            args.pair_weight < 0 or args.coverage_weight < 0 or
            args.target_pair_quota < 0 or args.target_pair_quota > args.max_pairs):
        raise ValueError("invalid selection hyperparameters")
    if (args.geometric_coverage_weight < 0 or args.geometric_coverage_k < 1 or
            not 0 <= args.geometric_coverage_tolerance <= 1):
        raise ValueError("invalid geometric coverage hyperparameters")
    if (args.integrated_coverage_weight < 0 or
            not 0 <= args.integrated_coverage_tolerance <= 1 or
            args.integrated_coverage_shortlist < 1 or
            args.integrated_coverage_folds < 1):
        raise ValueError("invalid integrated coverage hyperparameters")
    if (args.integrated_coverage_weight > 0 and
            args.integrated_coverage_shortlist > args.draws):
        raise ValueError("integrated coverage shortlist exceeds draws")
    if args.mode in {"consensus", "consensus_rake"}:
        if (not args.prior_index or len(args.prior_index) != len(args.prior_weight) or
                any(value <= 0 for value in args.prior_weight) or args.consensus_strength <= 0):
            raise ValueError("consensus mode requires matched positive priors")
    elif args.prior_index or args.prior_weight:
        raise ValueError("prior indices are only valid in consensus modes")
    args.output.mkdir(parents=True)
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_path = args.repo / "synthetic" / args.dataset / "real.csv"
    train = canonicalize(pd.read_csv(train_path), info)
    pool = canonicalize(pd.read_csv(args.pool), info)
    backbone_pool = canonicalize(pd.read_csv(args.backbone_pool), info)
    if len(pool) != len(backbone_pool) or len(pool) <= args.count:
        raise ValueError("pool/backbone size mismatch or no oversampling")
    include_pairs = args.mode in {
        "uniform_search", "rake_pair", "rake_marginal_pair_select",
        "hybrid_logit", "hybrid_xgb", "consensus", "consensus_rake"}
    groups = build_groups(
        train, pool, info, args.bins, args.pair_bins,
        args.max_pairs, include_pairs, args.target_pair_quota)
    weights = np.ones(len(pool), dtype=float)
    density_diagnostics: dict = {"classifier": None}
    if args.mode in {"consensus", "consensus_rake"}:
        votes = np.zeros(len(pool), dtype=float)
        prior_records = []
        for path, prior_weight in zip(args.prior_index, args.prior_weight):
            frame = pd.read_csv(path)
            if list(frame.columns) != ["pool_index"]:
                raise ValueError(f"invalid prior-index schema: {path}")
            indices = frame["pool_index"].to_numpy(np.int64)
            if (len(indices) != args.count or len(np.unique(indices)) != len(indices) or
                    indices.min(initial=0) < 0 or indices.max(initial=-1) >= len(pool)):
                raise ValueError(f"invalid prior indices: {path}")
            votes[indices] += prior_weight
            prior_records.append({
                "path": str(path), "sha256": sha256(path),
                "weight": prior_weight, "rows": len(indices),
            })
        votes /= float(sum(args.prior_weight))
        centered = votes - float(votes.mean())
        weights = np.exp(np.clip(args.consensus_strength * centered, -12.0, 12.0))
        weights /= weights.mean()
        weights = np.clip(weights, 1.0 / args.weight_clip, args.weight_clip)
        density_diagnostics = {
            "classifier": "consensus_prior", "priors": prior_records,
            "vote_min": float(votes.min()), "vote_max": float(votes.max()),
            "vote_mean": float(votes.mean()),
        }
    if args.mode in {"ratio_logit", "hybrid_logit"}:
        weights, density_diagnostics = density_ratio(
            train, pool, info, "logit", args.seed, args.weight_clip)
    elif args.mode in {"ratio_xgb", "hybrid_xgb"}:
        weights, density_diagnostics = density_ratio(
            train, pool, info, "xgb", args.seed, args.weight_clip)
    rake_trace: list[dict] = []
    if args.mode in {"rake_marginal", "rake_pair", "rake_marginal_pair_select", "hybrid_logit", "hybrid_xgb", "consensus_rake"}:
        rake_groups = groups
        if args.mode in {"rake_marginal_pair_select", "consensus_rake"}:
            rake_groups = [group for group in groups if group.family == "univariate"]
        weights, rake_trace = rake(
            weights, rake_groups, args.rake_iterations,
            args.rake_damping, args.weight_clip)
    geometry = None
    if args.geometric_coverage_weight > 0:
        geometry = build_geometry_coverage(
            train, pool, info, args.count, args.geometric_coverage_k)
    integrated = None
    if args.integrated_coverage_weight > 0:
        integrated = build_integrated_coverage(
            train, pool, info, args.count, args.seed,
            args.integrated_coverage_folds)
    selected, selection = sample_best(
        weights, groups, args.count, args.draws, args.seed,
        args.tail_mix, args.temperature, args.pair_weight,
        args.coverage_weight, geometry,
        args.geometric_coverage_weight,
        args.geometric_coverage_tolerance, integrated,
        args.integrated_coverage_weight,
        args.integrated_coverage_tolerance,
        args.integrated_coverage_shortlist)
    uniform = np.asarray(selection.pop("uniform_indices"), dtype=np.int64)
    write_csv(pool.iloc[selected].reset_index(drop=True), args.output / "selected.csv")
    write_csv(pool.iloc[:args.count].reset_index(drop=True), args.output / "g_reference.csv")
    write_csv(backbone_pool.iloc[:args.count].reset_index(drop=True), args.output / "b_reference.csv")
    write_csv(pool.iloc[uniform].reset_index(drop=True), args.output / "uniform_control.csv")
    pd.DataFrame({"pool_index": selected}).to_csv(args.output / "selected_indices.csv", index=False)
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "mode": args.mode,
        "seed": args.seed,
        "train_rows": len(train),
        "pool_rows": len(pool),
        "selected_rows": len(selected),
        "pool_sha256": sha256(args.pool),
        "backbone_pool_sha256": sha256(args.backbone_pool),
        "train_sha256": sha256(train_path),
        "parameters": {
            "bins": args.bins, "pair_bins": args.pair_bins,
            "max_pairs": args.max_pairs,
            "target_pair_quota": args.target_pair_quota,
            "pair_weight": args.pair_weight,
            "coverage_weight": args.coverage_weight,
            "consensus_strength": args.consensus_strength,
            "geometric_coverage_weight": args.geometric_coverage_weight,
            "geometric_coverage_k": args.geometric_coverage_k,
            "geometric_coverage_tolerance": args.geometric_coverage_tolerance,
            "integrated_coverage_weight": args.integrated_coverage_weight,
            "integrated_coverage_tolerance": args.integrated_coverage_tolerance,
            "integrated_coverage_shortlist": args.integrated_coverage_shortlist,
            "integrated_coverage_folds": args.integrated_coverage_folds,
            "rake_iterations": args.rake_iterations,
            "rake_damping": args.rake_damping,
            "weight_clip": args.weight_clip,
            "temperature": args.temperature,
            "tail_mix": args.tail_mix, "draws": args.draws,
        },
        "groups": [
            {"name": group.name, "family": group.family,
             "cardinality": group.cardinality} for group in groups],
        "density_ratio": density_diagnostics,
        "rake_trace": rake_trace,
        "selection": selection,
        "geometry_coverage": ({
            "enabled": True,
            "reference_score": geometry.reference_score,
            "selected_score": selection["best"]["geometric_coverage"],
            "selected_deficit": selection["best"]["geometric_coverage_deficit"],
        } if geometry else {"enabled": False}),
        "integrated_coverage": ({
            "enabled": True,
            "reference_score": integrated.reference_score,
            "selected_score": selection["best"]["integrated_coverage"],
            "selected_alpha": selection["best"]["integrated_alpha"],
            "selected_deficit": selection["best"]["integrated_coverage_deficit"],
            "fold_seeds": list(integrated.fold_seeds),
        } if integrated else {"enabled": False}),
        "weight_summary": {
            "min": float(weights.min()), "max": float(weights.max()),
            "mean": float(weights.mean()), "std": float(weights.std()),
        },
        "no_duplicate_selected_rows": len(np.unique(selected)) == len(selected),
    }
    (args.output / "selection_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output / "selected.csv")
    print(json.dumps({
        "best_objective": selection["best"]["objective"],
        "uniform_objective": selection["uniform_control"]["objective"],
        "effective_sample_size": selection["effective_sample_size"],
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
