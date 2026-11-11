#!/usr/bin/env python3
"""Train-only batch transport for distribution-aware diffusion guidance.

The operator is deliberately evaluator independent.  It uses empirical
training marginals and a Gaussian-copula correlation target, never held-out
rows or final evaluation metrics.  It can be used as a cheap post-generation
ceiling study and, through :mod:`dense_numeric_factor_messages`, as a bounded
clean-preview intervention at a middle reverse-diffusion state.

``marginal`` preserves the generated row ordering and transports each selected
numeric column toward its training empirical marginal.  ``copula`` first
aligns the Gaussian-rank correlation of the selected block and then maps the
result back through the training marginals.  ``graph_copula`` replaces only
correlations supported by verified graph adjacency; non-graph correlations
retain their current batch values before projection to the nearest PSD
correlation matrix.  ``adaptive_copula`` additionally activates the largest
current train--preview correlation gaps, turning the statistical graph into a
state-dependent object instead of a frozen list of edges.  ``sliced`` matches
train projections along several fixed multivariate directions, while
``graph_sliced`` restricts those directions to verified graph components and
``adaptive_sliced`` activates only the directions with the largest current
Wasserstein discrepancy.  The latter two make satisfied as well as violated
edges carry positive distributional information.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri


def edge_columns(edge: dict) -> set[str]:
    columns = set(edge.get("columns", []))
    program = edge.get("program", {})
    kind = program.get("kind")
    if kind in {"ordered_triplet", "sum_equal", "zero_cooccurrence"}:
        columns.update(program.get("columns", []))
    elif kind == "abs_equal":
        columns.update([program.get("source"), program.get("target_column")])
    elif kind == "less_equal":
        columns.update([program.get("left"), program.get("right")])
    for atom in edge.get("applicability", []) + edge.get("requirements", []):
        columns.add(atom.get("left_column"))
        if atom.get("right_kind") == "column":
            columns.add(atom.get("right_column"))
    return {column for column in columns if column}


def stable_gaussian_ranks(values: np.ndarray) -> np.ndarray:
    """Return finite normal scores with deterministic tie handling."""
    n, d = values.shape
    result = np.empty((n, d), dtype=np.float64)
    probabilities = (np.arange(n, dtype=np.float64) + 0.5) / n
    normal_scores = ndtri(probabilities)
    for column in range(d):
        order = np.argsort(values[:, column], kind="mergesort")
        result[order, column] = normal_scores
    return result


def correlation(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 1:
        return np.ones((1, 1), dtype=np.float64)
    output = np.corrcoef(values, rowvar=False)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def symmetric_power(matrix: np.ndarray, power: float, floor: float = 1e-4) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return (vectors * (values ** power)) @ vectors.T


def nearest_correlation(matrix: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    positive = symmetric_power(matrix, 1.0, floor=floor)
    scale = np.sqrt(np.maximum(np.diag(positive), floor))
    result = positive / np.outer(scale, scale)
    np.fill_diagonal(result, 1.0)
    return result


def empirical_quantiles(sorted_reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    order = np.argsort(scores, kind="mergesort")
    probabilities = (np.arange(n, dtype=np.float64) + 0.5) / n
    reference_probability = (np.arange(len(sorted_reference), dtype=np.float64) + 0.5) / len(sorted_reference)
    values = np.interp(probabilities, reference_probability, sorted_reference)
    output = np.empty(n, dtype=np.float64)
    output[order] = values
    return output


class BatchDistributionTransport:
    """Fit a reusable numeric transport using training rows only."""

    def __init__(self, train: pd.DataFrame, numeric_columns: list[str], edges: list[dict],
                 scope: str = "graph", mode: str = "graph_copula",
                 adaptive_top_k: int = 0, adaptive_min_gap: float = 0.0,
                 preserve_tail_fraction: float = 0.0,
                 sliced_directions: int = 32, seed: int = 20261019):
        if scope not in {"graph", "all"}:
            raise ValueError(f"unsupported scope: {scope}")
        if mode not in {"marginal", "copula", "pearson_copula", "graph_copula",
                        "adaptive_copula", "sliced", "graph_sliced", "adaptive_sliced"}:
            raise ValueError(f"unsupported transport mode: {mode}")
        graph_columns = set().union(*(edge_columns(edge) for edge in edges)) if edges else set()
        selected = numeric_columns if scope == "all" else [c for c in numeric_columns if c in graph_columns]
        self.columns = [column for column in selected if column in train]
        if not self.columns:
            raise ValueError("batch transport selected no numeric columns")
        self.mode = mode
        self.adaptive_top_k = max(int(adaptive_top_k), 0)
        self.adaptive_min_gap = max(float(adaptive_min_gap), 0.0)
        self.sliced_directions = max(int(sliced_directions), 1)
        self.seed = int(seed)
        if not 0 <= preserve_tail_fraction < 1:
            raise ValueError("preserve_tail_fraction must lie in [0,1)")
        self.preserve_tail_fraction = float(preserve_tail_fraction)
        matrix = train[self.columns].apply(pd.to_numeric, errors="coerce")
        self.medians = matrix.median().to_numpy(np.float64)
        values = matrix.fillna(pd.Series(dict(zip(self.columns, self.medians)))).to_numpy(np.float64)
        self.reference_values = values
        self.sorted_reference = [np.sort(values[:, index], kind="mergesort")
                                 for index in range(values.shape[1])]
        self.target_correlation = correlation(stable_gaussian_ranks(values))
        self.target_pearson_correlation = correlation(values)
        self.adjacency = np.eye(len(self.columns), dtype=bool)
        positions = {column: index for index, column in enumerate(self.columns)}
        for edge in edges:
            local = [positions[column] for column in edge_columns(edge) if column in positions]
            for left in local:
                for right in local:
                    self.adjacency[left, right] = True
        self.robust_center = np.median(values, axis=0)
        lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
        self.robust_scale = upper - lower
        fallback = np.std(values, axis=0)
        self.robust_scale = np.where(self.robust_scale > 1e-8, self.robust_scale,
                                     np.where(fallback > 1e-8, fallback, 1.0))
        self.reference_standardized = (
            (values - self.robust_center) / self.robust_scale)
        self.sliced_vectors = self._make_sliced_vectors(edges, positions)

    def _make_sliced_vectors(self, edges: list[dict], positions: dict[str, int]) -> np.ndarray:
        """Create deterministic all-column or graph-local projection vectors."""
        rng = np.random.default_rng(self.seed)
        dimension = len(self.columns)
        vectors: list[np.ndarray] = []
        graph_local = self.mode in {"graph_sliced", "adaptive_sliced"}
        supports: list[list[int]] = []
        if graph_local:
            for edge in edges:
                support = sorted({positions[column] for column in edge_columns(edge)
                                  if column in positions})
                if len(support) >= 2 and support not in supports:
                    supports.append(support)
        if not supports:
            supports = [list(range(dimension))]
        for ordinal in range(self.sliced_directions):
            support = supports[ordinal % len(supports)] if graph_local else supports[0]
            vector = np.zeros(dimension, dtype=np.float64)
            vector[support] = rng.normal(size=len(support))
            norm = np.linalg.norm(vector)
            if norm < 1e-12:
                vector[support[0]] = 1.0
                norm = 1.0
            vectors.append(vector / norm)
        return np.stack(vectors)

    def apply(self, frame: pd.DataFrame, strength: float) -> tuple[pd.DataFrame, np.ndarray, dict]:
        if not 0 <= strength <= 1:
            raise ValueError("transport strength must lie in [0,1]")
        result = frame.copy()
        current_frame = result[self.columns].apply(pd.to_numeric, errors="coerce")
        current = current_frame.fillna(
            pd.Series(dict(zip(self.columns, self.medians)))).to_numpy(np.float64)
        sliced_before = sliced_after = None
        active_sliced_directions = 0
        if self.mode in {"sliced", "graph_sliced", "adaptive_sliced"} and len(self.columns) > 1:
            standardized = (current - self.robust_center) / self.robust_scale
            reference = self.reference_standardized
            candidates = []
            for ordinal, vector in enumerate(self.sliced_vectors):
                projection = standardized @ vector
                reference_projection = np.sort(reference @ vector, kind="mergesort")
                target = empirical_quantiles(reference_projection, projection)
                gap = float(np.mean(np.abs(target - projection)))
                if gap >= self.adaptive_min_gap:
                    candidates.append((ordinal, gap))
            if self.mode == "adaptive_sliced" and self.adaptive_top_k > 0:
                candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[
                    :self.adaptive_top_k]
            # Largest-gap-first makes every finite compute budget target the
            # most visible state-specific distribution error.
            candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
            sliced_before = float(np.mean([gap for _, gap in candidates])) if candidates else 0.0
            transported_standardized = standardized.copy()
            for ordinal, _ in candidates:
                vector = self.sliced_vectors[ordinal]
                projection = transported_standardized @ vector
                reference_projection = np.sort(reference @ vector, kind="mergesort")
                target = empirical_quantiles(reference_projection, projection)
                transported_standardized += (target - projection)[:, None] * vector[None, :]
            transported = (
                self.robust_center + transported_standardized * self.robust_scale)
            after_gaps = []
            for ordinal, _ in candidates:
                vector = self.sliced_vectors[ordinal]
                projection = transported_standardized @ vector
                reference_projection = np.sort(reference @ vector, kind="mergesort")
                target = empirical_quantiles(reference_projection, projection)
                after_gaps.append(float(np.mean(np.abs(target - projection))))
            sliced_after = float(np.mean(after_gaps)) if after_gaps else 0.0
            active_sliced_directions = len(candidates)
        elif self.mode == "marginal" or len(self.columns) == 1:
            scores = current
            desired_correlation = correlation(stable_gaussian_ranks(current))
        elif self.mode == "pearson_copula":
            scale = np.std(current, axis=0)
            scale[scale < 1e-8] = 1.0
            standardized = (current - np.mean(current, axis=0)) / scale
            source_correlation = nearest_correlation(correlation(standardized))
            desired_correlation = nearest_correlation(self.target_pearson_correlation)
            transform = symmetric_power(source_correlation, -0.5) @ symmetric_power(
                desired_correlation, 0.5)
            scores = standardized @ transform
        else:
            gaussian = stable_gaussian_ranks(current)
            source_correlation = nearest_correlation(correlation(gaussian))
            active_adjacency = self.adjacency.copy()
            if self.mode == "adaptive_copula":
                gap = np.abs(source_correlation - self.target_correlation)
                upper = np.triu_indices(len(self.columns), k=1)
                eligible = np.flatnonzero(gap[upper] >= self.adaptive_min_gap)
                if self.adaptive_top_k > 0 and len(eligible) > self.adaptive_top_k:
                    order = np.argsort(gap[upper][eligible], kind="mergesort")
                    eligible = eligible[order[-self.adaptive_top_k:]]
                active_adjacency = self.adjacency.copy()
                left, right = upper[0][eligible], upper[1][eligible]
                active_adjacency[left, right] = True
                active_adjacency[right, left] = True
            if self.mode in {"graph_copula", "adaptive_copula"}:
                desired_correlation = source_correlation.copy()
                desired_correlation[active_adjacency] = self.target_correlation[active_adjacency]
                desired_correlation = nearest_correlation(desired_correlation)
            else:
                desired_correlation = nearest_correlation(self.target_correlation)
            transform = symmetric_power(source_correlation, -0.5) @ symmetric_power(
                desired_correlation, 0.5)
            scores = gaussian @ transform
        if self.mode not in {"sliced", "graph_sliced", "adaptive_sliced"} or len(self.columns) == 1:
            transported = np.column_stack([
                empirical_quantiles(self.sorted_reference[index], scores[:, index])
                for index in range(len(self.columns))
            ])
        updated = current + strength * (transported - current)
        protected_rows = np.zeros(len(current), dtype=bool)
        if self.preserve_tail_fraction > 0 and len(current) > 2:
            gaussian_rarity = np.mean(stable_gaussian_ranks(current) ** 2, axis=1)
            threshold = np.quantile(gaussian_rarity, 1.0 - self.preserve_tail_fraction)
            protected_rows = gaussian_rarity >= threshold
            updated[protected_rows] = current[protected_rows]
        changed_local = np.abs(updated - current) > (
            16 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(current)))
        result.loc[:, self.columns] = updated
        if self.mode == "pearson_copula":
            after_correlation = correlation(updated)
            before_correlation = correlation(current)
            diagnostic_target = self.target_pearson_correlation
        else:
            after_correlation = correlation(stable_gaussian_ranks(updated))
            before_correlation = correlation(stable_gaussian_ranks(current))
            diagnostic_target = self.target_correlation
        selected = (active_adjacency if self.mode in {"graph_copula", "adaptive_copula"}
                    else np.ones_like(self.adjacency, dtype=bool))
        off_diagonal = selected & ~np.eye(len(self.columns), dtype=bool)
        if off_diagonal.any():
            before_error = float(np.mean(np.abs(
                before_correlation[off_diagonal] - diagnostic_target[off_diagonal])))
            after_error = float(np.mean(np.abs(
                after_correlation[off_diagonal] - diagnostic_target[off_diagonal])))
        else:
            before_error = after_error = 0.0
        diagnostics = {
            "transport_columns": len(self.columns),
            "transport_cell_events": int(changed_local.sum()),
            "transport_row_events": int(changed_local.any(axis=1).sum()),
            "transport_correlation_error_before": before_error,
            "transport_correlation_error_after": after_error,
            "transport_mean_absolute_move": float(np.mean(np.abs(updated - current))),
            "transport_active_edges": (
                active_sliced_directions if self.mode in {"sliced", "graph_sliced", "adaptive_sliced"}
                else int(np.triu(selected, k=1).sum())),
            "transport_tail_rows_preserved": int(protected_rows.sum()),
        }
        if sliced_before is not None:
            diagnostics["transport_projection_error_before"] = sliced_before
            diagnostics["transport_projection_error_after"] = sliced_after
        return result, changed_local, diagnostics


class ConditionalBatchDistributionTransport:
    """Apply separate train-fitted transports inside categorical subgroups.

    Conditioning on the task label is especially useful for classification:
    it improves feature--target dependence without rewriting the generated
    labels.  During sampling the predicted category can change, so subgroup
    membership and therefore the active transport graph are row- and
    timestep-dependent.
    """

    def __init__(self, train: pd.DataFrame, numeric_columns: list[str], edges: list[dict],
                 condition_column: str, scope: str = "all", mode: str = "copula",
                 adaptive_top_k: int = 0, adaptive_min_gap: float = 0.0,
                 min_group_rows: int = 64, preserve_tail_fraction: float = 0.0,
                 sliced_directions: int = 32, seed: int = 20261019):
        if condition_column not in train:
            raise ValueError(f"condition column not found: {condition_column}")
        self.condition_column = condition_column
        self.global_operator = BatchDistributionTransport(
            train, numeric_columns, edges, scope=scope, mode=mode,
            adaptive_top_k=adaptive_top_k, adaptive_min_gap=adaptive_min_gap,
            preserve_tail_fraction=preserve_tail_fraction,
            sliced_directions=sliced_directions, seed=seed)
        self.columns = self.global_operator.columns
        self.operators: dict[str, BatchDistributionTransport] = {}
        for value, group in train.groupby(train[condition_column].astype(str), dropna=False):
            if len(group) < min_group_rows:
                continue
            self.operators[str(value).strip()] = BatchDistributionTransport(
                group, numeric_columns, edges, scope=scope, mode=mode,
                adaptive_top_k=adaptive_top_k, adaptive_min_gap=adaptive_min_gap,
                preserve_tail_fraction=preserve_tail_fraction,
                sliced_directions=sliced_directions, seed=seed)

    def apply(self, frame: pd.DataFrame, strength: float) -> tuple[pd.DataFrame, np.ndarray, dict]:
        result = frame.copy()
        changed = np.zeros((len(frame), len(self.columns)), dtype=bool)
        totals: dict[str, float] = {}
        groups_used = 0
        labels = frame[self.condition_column].astype(str).str.strip()
        for value, indices in labels.groupby(labels, sort=False).groups.items():
            operator = self.operators.get(str(value), self.global_operator)
            local_frame, local_changed, diagnostics = operator.apply(
                frame.loc[indices].copy(), strength)
            result.loc[indices, self.columns] = local_frame[self.columns].to_numpy()
            positions = frame.index.get_indexer(indices)
            changed[positions] = local_changed
            weight = len(indices)
            for key, metric in diagnostics.items():
                if key in {"transport_columns", "transport_active_edges"}:
                    totals[key] = totals.get(key, 0.0) + float(metric)
                elif key in {"transport_cell_events", "transport_row_events",
                             "transport_tail_rows_preserved"}:
                    totals[key] = totals.get(key, 0.0) + float(metric)
                else:
                    totals[key] = totals.get(key, 0.0) + weight * float(metric)
            groups_used += 1
        n = max(len(frame), 1)
        for key in list(totals):
            if key not in {"transport_columns", "transport_active_edges",
                           "transport_cell_events", "transport_row_events",
                           "transport_tail_rows_preserved"}:
                totals[key] /= n
        totals["transport_columns"] = len(self.columns)
        totals["transport_condition_groups"] = groups_used
        totals["transport_active_edges"] = int(totals.get("transport_active_edges", 0))
        totals["transport_cell_events"] = int(totals.get("transport_cell_events", 0))
        totals["transport_row_events"] = int(totals.get("transport_row_events", 0))
        totals["transport_tail_rows_preserved"] = int(
            totals.get("transport_tail_rows_preserved", 0))
        return result, changed, totals


class QuantileConditionalBatchDistributionTransport:
    """State-dependent transport inside train-fitted numerical quantile bins.

    For regression, the current clean prediction of the target selects a local
    train distribution.  Consequently the active conditional factor changes
    with both row and reverse state, while all bin boundaries and transports
    remain fitted on training data only.
    """

    def __init__(self, train: pd.DataFrame, numeric_columns: list[str], edges: list[dict],
                 condition_column: str, quantile_bins: int = 8,
                 scope: str = "all", mode: str = "copula",
                 adaptive_top_k: int = 0, adaptive_min_gap: float = 0.0,
                 min_group_rows: int = 64, preserve_tail_fraction: float = 0.0,
                 sliced_directions: int = 32, seed: int = 20261019):
        if condition_column not in train:
            raise ValueError(f"condition column not found: {condition_column}")
        self.condition_column = condition_column
        self.global_operator = BatchDistributionTransport(
            train, numeric_columns, edges, scope=scope, mode=mode,
            adaptive_top_k=adaptive_top_k, adaptive_min_gap=adaptive_min_gap,
            preserve_tail_fraction=preserve_tail_fraction,
            sliced_directions=sliced_directions, seed=seed)
        self.columns = self.global_operator.columns
        values = pd.to_numeric(train[condition_column], errors="coerce")
        fill = float(values.median())
        values = values.fillna(fill).to_numpy(float)
        requested = max(int(quantile_bins), 2)
        boundaries = np.unique(np.quantile(values, np.linspace(0, 1, requested + 1)))
        if len(boundaries) < 3:
            raise ValueError("numeric condition has fewer than two quantile bins")
        self.boundaries = boundaries
        labels = np.digitize(values, boundaries[1:-1], right=True)
        self.operators: dict[int, BatchDistributionTransport] = {}
        for label in np.unique(labels):
            group = train.iloc[np.flatnonzero(labels == label)]
            if len(group) < min_group_rows:
                continue
            self.operators[int(label)] = BatchDistributionTransport(
                group, numeric_columns, edges, scope=scope, mode=mode,
                adaptive_top_k=adaptive_top_k, adaptive_min_gap=adaptive_min_gap,
                preserve_tail_fraction=preserve_tail_fraction,
                sliced_directions=sliced_directions, seed=seed + int(label) + 1)

    def apply(self, frame: pd.DataFrame, strength: float) -> tuple[pd.DataFrame, np.ndarray, dict]:
        result = frame.copy()
        changed = np.zeros((len(frame), len(self.columns)), dtype=bool)
        totals: dict[str, float] = {}
        condition = pd.to_numeric(frame[self.condition_column], errors="coerce")
        condition = condition.fillna(float(np.median(self.boundaries)))
        labels = np.digitize(condition.to_numpy(float), self.boundaries[1:-1], right=True)
        groups_used = 0
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            operator = self.operators.get(int(label), self.global_operator)
            local_frame, local_changed, diagnostics = operator.apply(
                frame.iloc[positions].copy(), strength)
            result.iloc[positions, result.columns.get_indexer(self.columns)] = (
                local_frame[self.columns].to_numpy())
            changed[positions] = local_changed
            weight = len(positions)
            for key, metric in diagnostics.items():
                if key in {"transport_columns", "transport_active_edges",
                           "transport_cell_events", "transport_row_events",
                           "transport_tail_rows_preserved"}:
                    totals[key] = totals.get(key, 0.0) + float(metric)
                else:
                    totals[key] = totals.get(key, 0.0) + weight * float(metric)
            groups_used += 1
        n = max(len(frame), 1)
        additive = {"transport_columns", "transport_active_edges",
                    "transport_cell_events", "transport_row_events",
                    "transport_tail_rows_preserved"}
        for key in list(totals):
            if key not in additive:
                totals[key] /= n
        totals["transport_columns"] = len(self.columns)
        totals["transport_condition_groups"] = groups_used
        for key in additive - {"transport_columns"}:
            totals[key] = int(totals.get(key, 0))
        return result, changed, totals


def numeric_columns(info: dict) -> list[str]:
    positions = list(info["num_col_idx"])
    if info["task_type"] == "regression":
        positions = list(info["target_col_idx"]) + positions
    return [info["column_names"][position] for position in positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["marginal", "copula", "pearson_copula", "graph_copula", "adaptive_copula",
                                                   "sliced", "graph_sliced", "adaptive_sliced"],
                        default="graph_copula")
    parser.add_argument("--scope", choices=["graph", "all"], default="graph")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--adaptive-top-k", type=int, default=0)
    parser.add_argument("--adaptive-min-gap", type=float, default=0.0)
    parser.add_argument("--sliced-directions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--condition-column", default="",
                        help="Optional categorical column for subgroup-conditional transport.")
    parser.add_argument("--condition-quantile-bins", type=int, default=0,
                        help="If >1, condition on train-fitted numerical quantile bins instead of categorical values.")
    parser.add_argument("--preserve-tail-fraction", type=float, default=0.0,
                        help="Leave this high-rarity fraction untouched to retain coverage.")
    args = parser.parse_args()
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train = pd.read_csv(args.repo / "synthetic" / args.dataset / "real.csv")
    sample = pd.read_csv(args.input)
    edges = json.loads(args.graph.read_text())
    if args.condition_column and args.condition_quantile_bins > 1:
        operator_class = QuantileConditionalBatchDistributionTransport
    elif args.condition_column:
        operator_class = ConditionalBatchDistributionTransport
    else:
        operator_class = BatchDistributionTransport
    operator_kwargs = dict(
        scope=args.scope, mode=args.mode, adaptive_top_k=args.adaptive_top_k,
        adaptive_min_gap=args.adaptive_min_gap,
        preserve_tail_fraction=args.preserve_tail_fraction,
        sliced_directions=args.sliced_directions, seed=args.seed)
    if args.condition_column:
        operator_kwargs["condition_column"] = args.condition_column
    if args.condition_quantile_bins > 1:
        operator_kwargs["quantile_bins"] = args.condition_quantile_bins
    operator = operator_class(train, numeric_columns(info), edges, **operator_kwargs)
    transported, _, diagnostics = operator.apply(sample, args.strength)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    transported.to_csv(args.output, index=False)
    metadata = {**diagnostics, "mode": args.mode, "scope": args.scope,
                "strength": args.strength, "columns": operator.columns,
                "uses_heldout": False, "uses_evaluation_metric": False}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
