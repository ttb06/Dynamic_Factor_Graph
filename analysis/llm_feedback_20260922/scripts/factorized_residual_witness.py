#!/usr/bin/env python3
"""Cross-fitted factorized residual-witness transport.

The transport is intentionally train-only and evaluator-agnostic.  It builds
small, auditable heads rather than a single real-vs-synthetic discriminator:

* marginal heads propose empirical-quantile corrections;
* target heads model a numeric feature conditional on a read-only target;
* pair heads model one numeric feature conditional on another.

Conditional heads are calibrated from out-of-fold residuals.  At sampling
time all heads inspect the same clean preview, proposals are intersected by a
per-factor row budget and a per-row factor budget, and conflicting proposals
for one cell are combined by score x cross-fit reliability.  No held-out row,
synthetic quality metric, or final sample is used to fit or select a head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


VALID_MODES = {
    "none",
    "marginal",
    "marginal_target",
    "marginal_pair",
    "marginal_target_pair",
}


def _edge_columns(edge: dict) -> set[str]:
    """Collect columns from both typed predicates and executable programs."""
    columns = set(edge.get("columns", []))
    program = edge.get("program", {}) or {}
    for key in ("columns",):
        columns.update(program.get(key, []) or [])
    for key in ("source", "target_column", "left", "right"):
        value = program.get(key)
        if value:
            columns.add(value)
    for atom in edge.get("applicability", []) + edge.get("requirements", []):
        left = atom.get("left_column")
        right = atom.get("right_column") if atom.get("right_kind") == "column" else None
        if left:
            columns.add(left)
        if right:
            columns.add(right)
    return columns


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 1e-8:
        scale = float(np.std(values))
    return max(scale, 1e-6)


def _rank_to_reference(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    """Map stable ranks in ``values`` to an empirical reference distribution."""
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(sorted_reference, dtype=np.float64)
    n = len(values)
    if n < 2 or len(reference) < 2:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    probability = (np.arange(n, dtype=np.float64) + 0.5) / n
    reference_probability = (
        np.arange(len(reference), dtype=np.float64) + 0.5
    ) / len(reference)
    mapped = np.interp(probability, reference_probability, reference)
    result = np.empty(n, dtype=np.float64)
    result[order] = mapped
    return result


@dataclass
class _PredictorEncoder:
    name: str
    numeric: bool
    fill_value: float
    categories: dict[str, int] | None = None

    @classmethod
    def fit(cls, name: str, values: pd.Series) -> "_PredictorEncoder":
        numeric = pd.to_numeric(values, errors="coerce")
        # Treat a column as numeric only when nearly every non-null value is
        # genuinely convertible.  This keeps string labels deterministic.
        observed = int(values.notna().sum())
        numeric_observed = int(numeric.notna().sum())
        if observed == 0 or numeric_observed >= max(1, int(0.98 * observed)):
            finite = numeric[np.isfinite(numeric)]
            fill = float(finite.median()) if len(finite) else 0.0
            return cls(name=name, numeric=True, fill_value=fill)
        labels = values.astype("string").fillna("<NA>")
        unique = sorted(str(value) for value in labels.unique())
        mapping = {value: index for index, value in enumerate(unique)}
        return cls(name=name, numeric=False, fill_value=-1.0, categories=mapping)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.name]
        if self.numeric:
            return pd.to_numeric(values, errors="coerce").fillna(
                self.fill_value
            ).to_numpy(np.float64)
        assert self.categories is not None
        labels = values.astype("string").fillna("<NA>")
        return labels.map(self.categories).fillna(self.fill_value).to_numpy(np.float64)


@dataclass
class ResidualWitnessHead:
    """One independently calibrated proposal factor."""

    name: str
    kind: str
    destination: str
    predictors: tuple[str, ...]
    reliability: float
    residual_scale: float
    sorted_reference: np.ndarray
    model: ExtraTreesRegressor | None = None
    encoders: tuple[_PredictorEncoder, ...] = ()
    validation_r2: float | None = None
    crossfit_folds: int = 0

    def _predictors(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.encoders:
            return np.empty((len(frame), 0), dtype=np.float64)
        return np.column_stack([encoder.transform(frame) for encoder in self.encoders])

    def propose(self, frame: pd.DataFrame, destination_fill: float) -> tuple[np.ndarray, np.ndarray]:
        current = pd.to_numeric(
            frame[self.destination], errors="coerce"
        ).fillna(destination_fill).to_numpy(np.float64)
        if self.kind == "marginal":
            proposal = _rank_to_reference(current, self.sorted_reference)
        else:
            if self.model is None:
                raise RuntimeError(f"conditional head {self.name} has no fitted model")
            center = self.model.predict(self._predictors(frame))
            current_residual = current - center
            proposal = center + _rank_to_reference(
                current_residual, self.sorted_reference
            )
        delta = proposal - current
        score = np.abs(delta) / max(float(self.residual_scale), 1e-8)
        return proposal, np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "destination": self.destination,
            "predictors": list(self.predictors),
            "reliability": float(self.reliability),
            "residual_scale": float(self.residual_scale),
            "validation_r2": (
                None if self.validation_r2 is None else float(self.validation_r2)
            ),
            "crossfit_folds": int(self.crossfit_folds),
            "reference_rows": int(len(self.sorted_reference)),
        }


class FactorizedResidualWitnessTransport:
    """Train-fitted factorized residual witness with sparse row routing.

    ``target_column`` is allowed as a predictor but is always excluded from
    ``columns`` and can therefore never be an intervention destination.
    ``top_rows_per_factor`` accepts either a fraction in ``(0, 1]`` or an
    absolute row count greater than one; non-positive values disable this
    budget.  The existing trajectory driver's encoded trust region is applied
    after :meth:`apply`, so this class only performs head-local clipping.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        numeric_columns: list[str],
        edges: list[dict],
        *,
        mode: str = "marginal_target_pair",
        target_column: str | None = None,
        trees: int = 32,
        min_r2: float = 0.02,
        max_pair_heads: int = 16,
        top_factors_per_row: int = 2,
        top_rows_per_factor: float = 0.15,
        min_score: float = 0.25,
        clip: float = 2.0,
        seed: int = 20261021,
        **_: Any,
    ):
        if mode not in VALID_MODES - {"none"}:
            raise ValueError(f"unsupported residual-witness mode: {mode}")
        if trees < 1:
            raise ValueError("trees must be positive")
        if max_pair_heads < 0:
            raise ValueError("max_pair_heads must be nonnegative")
        if top_factors_per_row < 1:
            raise ValueError("top_factors_per_row must be at least one")
        if top_rows_per_factor < 0:
            raise ValueError("top_rows_per_factor must be nonnegative")
        if min_score < 0 or clip <= 0:
            raise ValueError("min_score must be nonnegative and clip must be positive")
        if target_column and target_column not in train.columns:
            raise ValueError(f"target column {target_column!r} is absent from training data")

        self.mode = mode
        self.target_column = target_column
        self.trees = int(trees)
        self.min_r2 = float(min_r2)
        self.max_pair_heads = int(max_pair_heads)
        self.top_factors_per_row = int(top_factors_per_row)
        self.top_rows_per_factor = float(top_rows_per_factor)
        self.min_score = float(min_score)
        self.clip = float(clip)
        self.seed = int(seed)
        self.training_rows = int(len(train))
        self.heldout_used_for_fitting = False

        # Numeric target values can appear in numeric_columns under TabDiff's
        # regression preprocessing.  Explicit exclusion is the central
        # target-read-only invariant.
        self.columns = [
            name for name in numeric_columns
            if name in train.columns and name != target_column
        ]
        if not self.columns:
            raise ValueError("residual witness selected no numeric destination columns")
        numeric_train = train[self.columns].apply(pd.to_numeric, errors="coerce")
        self.destination_fill = {
            name: float(numeric_train[name].median())
            if np.isfinite(float(numeric_train[name].median())) else 0.0
            for name in self.columns
        }
        self.destination_scale = {
            name: _robust_scale(
                numeric_train[name].fillna(self.destination_fill[name]).to_numpy(np.float64)
            )
            for name in self.columns
        }

        self.heads: list[ResidualWitnessHead] = []
        if "marginal" in mode:
            self._fit_marginal_heads(numeric_train)
        if "target" in mode:
            if not target_column:
                raise ValueError(
                    "target-conditional residual witness requires a target column"
                )
            self._fit_target_heads(train)
            if not any(head.kind == "target" for head in self.heads):
                raise ValueError(
                    "target-conditional residual witness was requested but no "
                    "target head passed cross-fit reliability"
                )
        if "pair" in mode and self.max_pair_heads:
            self._fit_pair_heads(train, edges)
            if not any(head.kind == "pair" for head in self.heads):
                raise ValueError(
                    "pair residual witness was requested but no pair head passed "
                    "cross-fit reliability; lower --residual-witness-min-r2 or "
                    "use marginal_target for an explicit zero-pair arm"
                )
        if not self.heads:
            raise ValueError(
                "residual witness retained no heads; lower --residual-witness-min-r2 "
                "or choose a mode containing marginal"
            )

        self._previous_index: np.ndarray | None = None
        self._previous_selected: np.ndarray | None = None
        self._previous_dominant: np.ndarray | None = None

    def _fit_marginal_heads(self, numeric_train: pd.DataFrame) -> None:
        support_reliability = float(min(1.0, np.sqrt(max(len(numeric_train), 1)) / 50.0))
        for destination in self.columns:
            values = numeric_train[destination].fillna(
                self.destination_fill[destination]
            ).to_numpy(np.float64)
            if np.unique(values).size < 2:
                continue
            self.heads.append(ResidualWitnessHead(
                name=f"marginal:{destination}",
                kind="marginal",
                destination=destination,
                predictors=(),
                reliability=support_reliability,
                residual_scale=self.destination_scale[destination],
                sorted_reference=np.sort(values, kind="mergesort"),
            ))

    def _fold_count(self, n: int) -> int:
        if n >= 250:
            return 5
        if n >= 90:
            return 3
        return 2

    def _fit_conditional_head(
        self,
        train: pd.DataFrame,
        destination: str,
        predictors: Iterable[str],
        kind: str,
        ordinal: int,
    ) -> ResidualWitnessHead | None:
        predictors = tuple(predictors)
        if not predictors or any(name not in train for name in predictors):
            return None
        y_series = pd.to_numeric(train[destination], errors="coerce").fillna(
            self.destination_fill[destination]
        )
        y = y_series.to_numpy(np.float64)
        if len(y) < 40 or np.unique(y).size < 2:
            return None
        encoders = tuple(_PredictorEncoder.fit(name, train[name]) for name in predictors)
        x = np.column_stack([encoder.transform(train) for encoder in encoders])
        folds = self._fold_count(len(y))
        splitter = KFold(n_splits=folds, shuffle=True, random_state=self.seed + ordinal)
        oof = np.empty(len(y), dtype=np.float64)
        leaf = max(4, min(32, len(y) // 250))
        for fold, (fit_index, validation_index) in enumerate(splitter.split(x)):
            model = ExtraTreesRegressor(
                n_estimators=self.trees,
                min_samples_leaf=leaf,
                max_features=1.0,
                n_jobs=-1,
                random_state=self.seed + 1000 * ordinal + fold,
            )
            model.fit(x[fit_index], y[fit_index])
            oof[validation_index] = model.predict(x[validation_index])
        score = float(r2_score(y, oof))
        if not np.isfinite(score) or score < self.min_r2:
            return None
        residual = y - oof
        residual_scale = max(
            float(np.quantile(np.abs(residual), 0.75)),
            0.05 * self.destination_scale[destination],
            1e-6,
        )
        model = ExtraTreesRegressor(
            n_estimators=self.trees,
            min_samples_leaf=leaf,
            max_features=1.0,
            n_jobs=-1,
            random_state=self.seed + 1_000_000 + ordinal,
        )
        model.fit(x, y)
        return ResidualWitnessHead(
            name=f"{kind}:{destination}<-{'+'.join(predictors)}",
            kind=kind,
            destination=destination,
            predictors=predictors,
            reliability=float(np.clip(score, 0.0, 1.0)),
            residual_scale=residual_scale,
            sorted_reference=np.sort(residual, kind="mergesort"),
            model=model,
            encoders=encoders,
            validation_r2=score,
            crossfit_folds=folds,
        )

    def _fit_target_heads(self, train: pd.DataFrame) -> None:
        assert self.target_column is not None
        for index, destination in enumerate(self.columns):
            head = self._fit_conditional_head(
                train, destination, (self.target_column,), "target", 10_000 + index
            )
            if head is not None:
                self.heads.append(head)

    def _candidate_pairs(self, train: pd.DataFrame, edges: list[dict]) -> list[tuple[str, str]]:
        allowed = set(self.columns)
        graph_pairs: set[tuple[str, str]] = set()
        for edge in edges:
            local = sorted(_edge_columns(edge) & allowed)
            for left_index, left in enumerate(local):
                for right in local[left_index + 1:]:
                    graph_pairs.add((left, right))

        numeric = train[self.columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(pd.Series(self.destination_fill))
        correlation = numeric.corr(method="spearman").abs().fillna(0.0)
        ranked: list[tuple[float, str, str]] = []
        for left_index, left in enumerate(self.columns):
            for right in self.columns[left_index + 1:]:
                bonus = 2.0 if tuple(sorted((left, right))) in graph_pairs else 0.0
                ranked.append((bonus + float(correlation.loc[left, right]), left, right))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        # Fit both directions for a bounded candidate set, then retain the
        # strongest cross-fitted directed heads.  Graph-supported pairs sort
        # ahead of correlation-only fallbacks without requiring graph coverage.
        candidate_pair_count = min(len(ranked), max(self.max_pair_heads, 1))
        return [(left, right) for _, left, right in ranked[:candidate_pair_count]]

    def _fit_pair_heads(self, train: pd.DataFrame, edges: list[dict]) -> None:
        candidates: list[ResidualWitnessHead] = []
        ordinal = 20_000
        for left, right in self._candidate_pairs(train, edges):
            for destination, predictor in ((left, right), (right, left)):
                head = self._fit_conditional_head(
                    train, destination, (predictor,), "pair", ordinal
                )
                ordinal += 1
                if head is not None:
                    candidates.append(head)
        candidates.sort(
            key=lambda head: (-head.reliability, head.destination, head.predictors)
        )
        self.heads.extend(candidates[:self.max_pair_heads])

    def _row_budget(self, n: int) -> int:
        if self.top_rows_per_factor <= 0:
            return n
        if self.top_rows_per_factor <= 1:
            return max(1, int(np.ceil(self.top_rows_per_factor * n)))
        return min(n, max(1, int(self.top_rows_per_factor)))

    def apply(self, frame: pd.DataFrame, strength: float) -> tuple[pd.DataFrame, np.ndarray, dict]:
        strength = float(np.clip(strength, 0.0, 1.0))
        result = frame.copy()
        result[self.columns] = result[self.columns].apply(
            pd.to_numeric, errors="coerce"
        ).astype(np.float64)
        n = len(result)
        h = len(self.heads)
        deltas = np.empty((h, n), dtype=np.float64)
        scores = np.empty((h, n), dtype=np.float64)
        destinations = {name: index for index, name in enumerate(self.columns)}

        for head_index, head in enumerate(self.heads):
            current = pd.to_numeric(
                frame[head.destination], errors="coerce"
            ).fillna(self.destination_fill[head.destination]).to_numpy(np.float64)
            proposal, raw_score = head.propose(
                frame, self.destination_fill[head.destination]
            )
            delta = np.clip(
                proposal - current,
                -self.clip * head.residual_scale,
                self.clip * head.residual_scale,
            )
            deltas[head_index] = delta
            scores[head_index] = raw_score * head.reliability

        eligible = scores >= self.min_score
        row_budget = self._row_budget(n)
        factor_selected = np.zeros_like(eligible)
        for head_index in range(h):
            candidates = np.flatnonzero(eligible[head_index])
            if len(candidates) > row_budget:
                order = np.argsort(
                    -scores[head_index, candidates], kind="mergesort"
                )[:row_budget]
                candidates = candidates[order]
            factor_selected[head_index, candidates] = True

        # Compute both sparse gates from the same unmodified eligible score
        # matrix.  The final router is their literal intersection; applying
        # one budget before ranking the other would couple the two controls.
        row_selected = np.zeros_like(factor_selected)
        for row in range(n):
            candidates = np.flatnonzero(eligible[:, row])
            if len(candidates) > self.top_factors_per_row:
                order = np.argsort(-scores[candidates, row], kind="mergesort")
                candidates = candidates[order[:self.top_factors_per_row]]
            row_selected[candidates, row] = True
        selected = factor_selected & row_selected

        weighted_delta = np.zeros((n, len(self.columns)), dtype=np.float64)
        weight_sum = np.zeros_like(weighted_delta)
        contributor_count = np.zeros_like(weighted_delta, dtype=np.int16)
        movement_by_kind = {"marginal": 0.0, "target": 0.0, "pair": 0.0}
        selected_by_kind = {"marginal": 0, "target": 0, "pair": 0}
        for head_index, head in enumerate(self.heads):
            rows = selected[head_index]
            if not rows.any():
                continue
            destination = destinations[head.destination]
            weight = scores[head_index, rows].clip(min=1e-12)
            weighted_delta[rows, destination] += deltas[head_index, rows] * weight
            weight_sum[rows, destination] += weight
            contributor_count[rows, destination] += 1
            selected_count = int(rows.sum())
            selected_by_kind[head.kind] += selected_count
            movement_by_kind[head.kind] += float(
                np.abs(deltas[head_index, rows]).sum()
            )

        nonzero_weight = weight_sum > 0
        aggregate_delta = np.zeros_like(weighted_delta)
        aggregate_delta[nonzero_weight] = (
            weighted_delta[nonzero_weight] / weight_sum[nonzero_weight]
        )
        aggregate_delta *= strength
        changed = np.abs(aggregate_delta) > (
            8 * np.finfo(np.float32).eps
            * np.maximum(
                1.0,
                np.abs(result[self.columns].to_numpy(np.float64)),
            )
        )
        for destination, column in destinations.items():
            rows = changed[:, column]
            if rows.any():
                values = pd.to_numeric(
                    result[destination], errors="coerce"
                ).fillna(self.destination_fill[destination]).to_numpy(
                    np.float64, copy=True
                )
                values[rows] += aggregate_delta[rows, column]
                result.loc[result.index[rows], destination] = values[rows]

        selected_scores = scores[selected]
        candidate_scores = scores[eligible]
        selected_rows = selected.any(axis=0)
        current_index = np.asarray(frame.index)
        overlap = 0.0
        switch_fraction = 0.0
        dominant = np.full(n, -1, dtype=np.int32)
        for row in range(n):
            candidates = np.flatnonzero(selected[:, row])
            if len(candidates):
                dominant[row] = int(candidates[np.argmax(scores[candidates, row])])
        if (
            self._previous_selected is not None
            and self._previous_selected.shape == selected.shape
            and self._previous_index is not None
            and np.array_equal(self._previous_index, current_index)
        ):
            union = np.logical_or(selected, self._previous_selected).sum(axis=0)
            intersection = np.logical_and(selected, self._previous_selected).sum(axis=0)
            valid = union > 0
            if valid.any():
                overlap = float(np.mean(intersection[valid] / union[valid]))
            assert self._previous_dominant is not None
            both = (dominant >= 0) & (self._previous_dominant >= 0)
            if both.any():
                switch_fraction = float(
                    np.mean(dominant[both] != self._previous_dominant[both])
                )
        self._previous_index = current_index.copy()
        self._previous_selected = selected.copy()
        self._previous_dominant = dominant.copy()

        target_preserved = 1
        if self.target_column and self.target_column in frame:
            # Series.equals handles both numerical NaNs and categorical/string
            # targets; np.array_equal(equal_nan=True) raises on object arrays.
            target_preserved = int(
                frame[self.target_column].reset_index(drop=True).equals(
                    result[self.target_column].reset_index(drop=True)
                )
            )

        row_factor_total = int(selected.sum())
        cell_events = int(changed.sum())
        diagnostics = {
            # Compatibility counters are aggregated by the trajectory driver.
            "transport_cell_events": cell_events,
            "transport_row_events": int(changed.any(axis=1).sum()),
            "transport_mean_absolute_move": float(np.abs(aggregate_delta).mean()),
            "residual_witness_heads_total": h,
            # Keep the non-scalar manifest outside the numeric
            # residual_witness_* namespace so generic CSV aggregators can
            # safely parse every prefixed field as a finite scalar.
            "witness_retained_head_ids": [
                head.name for head in self.heads
            ],
            "witness_retained_head_reliabilities": {
                head.name: float(head.reliability) for head in self.heads
            },
            "residual_witness_marginal_heads": sum(
                head.kind == "marginal" for head in self.heads
            ),
            "residual_witness_target_heads": sum(
                head.kind == "target" for head in self.heads
            ),
            "residual_witness_pair_heads": sum(
                head.kind == "pair" for head in self.heads
            ),
            "residual_witness_pair_budget_requested": self.max_pair_heads,
            "residual_witness_candidate_row_factors": int(eligible.sum()),
            "residual_witness_factor_budget_row_factors": int(factor_selected.sum()),
            "residual_witness_row_budget_row_factors": int(row_selected.sum()),
            "residual_witness_selected_row_factors": row_factor_total,
            "residual_witness_selected_rows": int(selected_rows.sum()),
            "residual_witness_selected_cells": cell_events,
            "residual_witness_selected_fraction": float(
                row_factor_total / max(n * h, 1)
            ),
            "residual_witness_candidate_score_mean": float(
                candidate_scores.mean() if len(candidate_scores) else 0.0
            ),
            "residual_witness_candidate_score_p90": float(
                np.quantile(candidate_scores, 0.90) if len(candidate_scores) else 0.0
            ),
            "residual_witness_selected_score_mean": float(
                selected_scores.mean() if len(selected_scores) else 0.0
            ),
            "residual_witness_selected_score_p90": float(
                np.quantile(selected_scores, 0.90) if len(selected_scores) else 0.0
            ),
            "residual_witness_conflicting_cells": int(
                (contributor_count > 1).sum()
            ),
            "residual_witness_head_overlap": overlap,
            "residual_witness_head_switch_fraction": switch_fraction,
            "residual_witness_marginal_selected_row_factors": selected_by_kind["marginal"],
            "residual_witness_target_selected_row_factors": selected_by_kind["target"],
            "residual_witness_pair_selected_row_factors": selected_by_kind["pair"],
            "residual_witness_marginal_proposed_absolute_move": movement_by_kind["marginal"],
            "residual_witness_target_proposed_absolute_move": movement_by_kind["target"],
            "residual_witness_pair_proposed_absolute_move": movement_by_kind["pair"],
            "residual_witness_reliability_mean": float(
                np.mean([head.reliability for head in self.heads])
            ),
            "residual_witness_target_preserved": target_preserved,
            "residual_witness_strength": strength,
        }
        return result, changed, diagnostics

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "target_read_only": True,
            "trees": self.trees,
            "min_r2": self.min_r2,
            "max_pair_heads": self.max_pair_heads,
            "top_factors_per_row": self.top_factors_per_row,
            "top_rows_per_factor": self.top_rows_per_factor,
            "min_score": self.min_score,
            "clip": self.clip,
            "seed": self.seed,
            "fit_source": "training_split_only",
            "training_rows": self.training_rows,
            "heldout_used_for_fitting": False,
            "heads": [head.metadata() for head in self.heads],
        }

