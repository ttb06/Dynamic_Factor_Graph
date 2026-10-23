#!/usr/bin/env python3
"""Typed factorized random-feature MMD transport for tabular previews.

The operator estimates a differentiable MMD witness independently for each
numeric marginal and selected numeric pair.  At a clean denoised preview it
routes the negative witness gradient only to the largest row-factor residuals.
The target is never an intervention destination, and all reference moments are
fit from the training split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


VALID_MODES = {"none", "marginal", "marginal_pair"}


def _edge_columns(edge: dict) -> set[str]:
    columns = set(edge.get("columns", []) or [])
    program = edge.get("program", {}) or {}
    columns.update(program.get("columns", []) or [])
    for key in ("source", "target_column", "left", "right"):
        value = program.get(key)
        if value:
            columns.add(value)
    for atom in (edge.get("applicability", []) or []) + (edge.get("requirements", []) or []):
        left = atom.get("left_column")
        right = atom.get("right_column") if atom.get("right_kind") == "column" else None
        if left:
            columns.add(left)
        if right:
            columns.add(right)
    return columns


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    location = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 1e-8:
        scale = float(np.std(values))
    return location, max(scale, 1e-6)


def _median_bandwidth(z: np.ndarray, seed: int) -> float:
    if len(z) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    if len(z) > 512:
        z = z[rng.choice(len(z), size=512, replace=False)]
    # A bounded random set of pairs avoids a quadratic fit-time allocation.
    pair_count = min(4096, max(256, len(z) * 4))
    left = rng.integers(0, len(z), size=pair_count)
    right = rng.integers(0, len(z), size=pair_count)
    distances = np.linalg.norm(z[left] - z[right], axis=1)
    distances = distances[np.isfinite(distances) & (distances > 1e-8)]
    return max(float(np.median(distances)) if len(distances) else 1.0, 0.10)


@dataclass
class RFFMMDHead:
    name: str
    kind: str
    columns: tuple[str, ...]
    reliability: float
    location: np.ndarray
    scale: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    reference_mean: np.ndarray
    bandwidth: float
    support_rows: int

    @property
    def feature_scale(self) -> float:
        return float(np.sqrt(2.0 / self.omega.shape[0]))

    def standardized(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, list(self.columns)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float64)
        values = np.where(np.isfinite(values), values, self.location)
        return np.clip((values - self.location) / self.scale, -8.0, 8.0)

    def witness(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        z = self.standardized(frame)
        angle = z @ self.omega.T + self.phase
        features = self.feature_scale * np.cos(angle)
        residual = features.mean(axis=0) - self.reference_mean
        energy = float(np.dot(residual, residual))
        # Negative gradient of ||mean(phi(x))-mean(phi(real))||^2, with the
        # common 2/n scalar omitted and restored through the bounded step size.
        descent = self.feature_scale * (
            (np.sin(angle) * residual[None, :]) @ self.omega
        )
        norm = np.linalg.norm(descent, axis=1)
        finite = norm[np.isfinite(norm)]
        normalizer = max(
            float(np.quantile(finite, 0.90)) if len(finite) else 0.0,
            1e-8,
        )
        score = np.nan_to_num(norm / normalizer, nan=0.0, posinf=0.0, neginf=0.0)
        return descent / normalizer, score * self.reliability, energy

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "columns": list(self.columns),
            "reliability": float(self.reliability),
            "bandwidth": float(self.bandwidth),
            "features": int(self.omega.shape[0]),
            "support_rows": int(self.support_rows),
        }


class FactorizedRFFMMDTransport:
    """Training-reference RFF-MMD guidance with sparse row-factor routing."""

    def __init__(
        self,
        train: pd.DataFrame,
        numeric_columns: list[str],
        edges: list[dict],
        *,
        mode: str = "marginal_pair",
        target_column: str | None = None,
        features: int = 64,
        bandwidth_scale: float = 1.0,
        max_pair_heads: int = 8,
        top_factors_per_row: int = 1,
        top_rows_per_factor: float = 0.15,
        min_score: float = 0.25,
        action_clip: float = 0.50,
        seed: int = 20261021,
        **_: Any,
    ):
        if mode not in VALID_MODES - {"none"}:
            raise ValueError(f"unsupported factor-MMD mode: {mode}")
        if features < 8:
            raise ValueError("factor-MMD requires at least eight random features")
        if bandwidth_scale <= 0 or action_clip <= 0:
            raise ValueError("bandwidth scale and action clip must be positive")
        if max_pair_heads < 0 or top_factors_per_row < 1 or top_rows_per_factor < 0:
            raise ValueError("invalid pair or routing budget")
        if min_score < 0:
            raise ValueError("minimum score must be nonnegative")

        self.mode = mode
        self.target_column = target_column
        self.features = int(features)
        self.bandwidth_scale = float(bandwidth_scale)
        self.max_pair_heads = int(max_pair_heads)
        self.top_factors_per_row = int(top_factors_per_row)
        self.top_rows_per_factor = float(top_rows_per_factor)
        self.min_score = float(min_score)
        self.action_clip = float(action_clip)
        self.seed = int(seed)
        self.heldout_used_for_fitting = False
        self.columns = [
            column for column in numeric_columns
            if column in train.columns and column != target_column
        ]
        if not self.columns:
            raise ValueError("factor-MMD selected no numeric destination columns")

        numeric = train[self.columns].apply(pd.to_numeric, errors="coerce")
        self.fill: dict[str, float] = {}
        self.location: dict[str, float] = {}
        self.scale: dict[str, float] = {}
        for column in self.columns:
            values = numeric[column].to_numpy(np.float64)
            location, scale = _robust_location_scale(values)
            self.fill[column] = location
            self.location[column] = location
            self.scale[column] = scale
            numeric[column] = numeric[column].fillna(location)

        self.heads: list[RFFMMDHead] = []
        self._fit_marginals(numeric)
        if "pair" in mode and self.max_pair_heads:
            self._fit_pairs(numeric, edges)
            if not any(head.kind == "pair" for head in self.heads):
                raise ValueError("pair MMD requested but no pair head was retained")
        self._previous_index: np.ndarray | None = None
        self._previous_selected: np.ndarray | None = None

    def _make_head(
        self,
        numeric: pd.DataFrame,
        columns: tuple[str, ...],
        kind: str,
        reliability: float,
        ordinal: int,
    ) -> RFFMMDHead:
        location = np.array([self.location[column] for column in columns], dtype=np.float64)
        scale = np.array([self.scale[column] for column in columns], dtype=np.float64)
        values = numeric.loc[:, list(columns)].to_numpy(np.float64)
        z = np.clip((values - location) / scale, -8.0, 8.0)
        bandwidth = _median_bandwidth(z, self.seed + ordinal) * self.bandwidth_scale
        rng = np.random.default_rng(self.seed + 10_000 * ordinal)
        omega = rng.normal(
            0.0, 1.0 / max(bandwidth, 1e-6),
            size=(self.features, len(columns)),
        )
        phase = rng.uniform(0.0, 2.0 * np.pi, size=self.features)
        reference_mean = np.sqrt(2.0 / self.features) * np.cos(
            z @ omega.T + phase
        ).mean(axis=0)
        return RFFMMDHead(
            name=f"{kind}:{'+'.join(columns)}",
            kind=kind,
            columns=columns,
            reliability=float(np.clip(reliability, 0.05, 1.0)),
            location=location,
            scale=scale,
            omega=omega,
            phase=phase,
            reference_mean=reference_mean,
            bandwidth=bandwidth,
            support_rows=len(numeric),
        )

    def _fit_marginals(self, numeric: pd.DataFrame) -> None:
        for ordinal, column in enumerate(self.columns, start=1):
            if numeric[column].nunique(dropna=False) < 2:
                continue
            self.heads.append(self._make_head(
                numeric, (column,), "marginal", 1.0, ordinal
            ))
        if not self.heads:
            raise ValueError("factor-MMD retained no marginal heads")

    def _fit_pairs(self, numeric: pd.DataFrame, edges: list[dict]) -> None:
        graph_pairs: set[tuple[str, str]] = set()
        allowed = set(self.columns)
        for edge in edges:
            local = sorted(_edge_columns(edge) & allowed)
            for left_index, left in enumerate(local):
                for right in local[left_index + 1:]:
                    graph_pairs.add((left, right))
        correlation = numeric.corr(method="spearman").abs().fillna(0.0)
        ranked: list[tuple[float, float, str, str]] = []
        for left_index, left in enumerate(self.columns):
            for right in self.columns[left_index + 1:]:
                corr = float(correlation.loc[left, right])
                graph = 1.0 if tuple(sorted((left, right))) in graph_pairs else 0.0
                ranked.append((graph, corr, left, right))
        ranked.sort(key=lambda value: (-value[0], -value[1], value[2], value[3]))
        for offset, (_, corr, left, right) in enumerate(
            ranked[: self.max_pair_heads], start=1
        ):
            self.heads.append(self._make_head(
                numeric,
                (left, right),
                "pair",
                max(corr, 0.10),
                1000 + offset,
            ))

    def _row_budget(self, rows: int) -> int:
        if self.top_rows_per_factor <= 0:
            return rows
        if self.top_rows_per_factor <= 1:
            return max(1, int(np.ceil(self.top_rows_per_factor * rows)))
        return min(rows, max(1, int(self.top_rows_per_factor)))

    def apply(self, frame: pd.DataFrame, strength: float) -> tuple[pd.DataFrame, np.ndarray, dict]:
        strength = float(np.clip(strength, 0.0, 1.0))
        result = frame.copy()
        current = result[self.columns].apply(pd.to_numeric, errors="coerce")
        for column in self.columns:
            current[column] = current[column].fillna(self.fill[column])
        result[self.columns] = current.astype(np.float64)

        rows = len(frame)
        head_count = len(self.heads)
        scores = np.zeros((head_count, rows), dtype=np.float64)
        head_deltas: list[np.ndarray] = []
        energies: list[float] = []
        descent_z: list[np.ndarray] = []
        for index, head in enumerate(self.heads):
            descent, score, energy = head.witness(frame)
            norm = np.linalg.norm(descent, axis=1)
            limiter = np.maximum(1.0, norm / self.action_clip)
            bounded = descent / limiter[:, None]
            delta = bounded * head.scale[None, :]
            scores[index] = score
            head_deltas.append(delta)
            energies.append(energy)
            descent_z.append(descent)

        eligible = scores >= self.min_score
        factor_selected = np.zeros_like(eligible)
        factor_budget = self._row_budget(rows)
        for head_index in range(head_count):
            candidates = np.flatnonzero(eligible[head_index])
            if len(candidates) > factor_budget:
                order = np.argsort(-scores[head_index, candidates], kind="mergesort")
                candidates = candidates[order[:factor_budget]]
            factor_selected[head_index, candidates] = True
        row_selected = np.zeros_like(eligible)
        for row in range(rows):
            candidates = np.flatnonzero(eligible[:, row])
            if len(candidates) > self.top_factors_per_row:
                order = np.argsort(-scores[candidates, row], kind="mergesort")
                candidates = candidates[order[: self.top_factors_per_row]]
            row_selected[candidates, row] = True
        selected = factor_selected & row_selected

        positions = {column: index for index, column in enumerate(self.columns)}
        weighted_delta = np.zeros((rows, len(self.columns)), dtype=np.float64)
        weight_sum = np.zeros_like(weighted_delta)
        contributor_count = np.zeros_like(weighted_delta, dtype=np.int16)
        selected_by_kind = {"marginal": 0, "pair": 0}
        predicted_decrease = 0.0
        for head_index, head in enumerate(self.heads):
            active = selected[head_index]
            if not active.any():
                continue
            weight = np.maximum(scores[head_index, active], 1e-12)
            local_delta = head_deltas[head_index][active]
            for local, column in enumerate(head.columns):
                position = positions[column]
                weighted_delta[active, position] += local_delta[:, local] * weight
                weight_sum[active, position] += weight
                contributor_count[active, position] += 1
            predicted_decrease += float(np.sum(
                descent_z[head_index][active]
                * (local_delta / head.scale[None, :])
            ))
            selected_by_kind[head.kind] += int(active.sum())

        nonzero = weight_sum > 0
        aggregate = np.zeros_like(weighted_delta)
        aggregate[nonzero] = weighted_delta[nonzero] / weight_sum[nonzero]
        aggregate *= strength
        original = result[self.columns].to_numpy(np.float64)
        changed = np.abs(aggregate) > (
            8 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(original))
        )
        updated = original + aggregate
        for position, column in enumerate(self.columns):
            active = changed[:, position]
            if active.any():
                result.loc[result.index[active], column] = updated[active, position]

        overlap = 0.0
        current_index = np.asarray(frame.index)
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
        self._previous_index = current_index.copy()
        self._previous_selected = selected.copy()

        target_preserved = 1
        if self.target_column and self.target_column in frame:
            target_preserved = int(
                frame[self.target_column].reset_index(drop=True).equals(
                    result[self.target_column].reset_index(drop=True)
                )
            )
        selected_scores = scores[selected]
        diagnostics = {
            "transport_cell_events": int(changed.sum()),
            "transport_row_events": int(changed.any(axis=1).sum()),
            "transport_mean_absolute_move": float(np.abs(aggregate).mean()),
            "factor_mmd_heads_total": head_count,
            "factor_mmd_marginal_heads": sum(head.kind == "marginal" for head in self.heads),
            "factor_mmd_pair_heads": sum(head.kind == "pair" for head in self.heads),
            "factor_mmd_energy_before": float(np.sum(energies)),
            "factor_mmd_candidate_row_factors": int(eligible.sum()),
            "factor_mmd_factor_budget_row_factors": int(factor_selected.sum()),
            "factor_mmd_row_budget_row_factors": int(row_selected.sum()),
            "factor_mmd_selected_row_factors": int(selected.sum()),
            "factor_mmd_selected_rows": int(selected.any(axis=0).sum()),
            "factor_mmd_selected_cells": int(changed.sum()),
            "factor_mmd_conflicting_cells": int((contributor_count > 1).sum()),
            "factor_mmd_selected_score_mean": float(selected_scores.mean() if len(selected_scores) else 0.0),
            "factor_mmd_selected_score_p90": float(np.quantile(selected_scores, 0.90) if len(selected_scores) else 0.0),
            "factor_mmd_marginal_selected_row_factors": selected_by_kind["marginal"],
            "factor_mmd_pair_selected_row_factors": selected_by_kind["pair"],
            "factor_mmd_predicted_energy_decrease": float(predicted_decrease * strength),
            "factor_mmd_head_overlap": overlap,
            "factor_mmd_target_preserved": target_preserved,
            "factor_mmd_strength": strength,
        }
        return result, changed, diagnostics

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "columns": list(self.columns),
            "target_column": self.target_column,
            "target_read_only": True,
            "features": self.features,
            "bandwidth_scale": self.bandwidth_scale,
            "max_pair_heads": self.max_pair_heads,
            "top_factors_per_row": self.top_factors_per_row,
            "top_rows_per_factor": self.top_rows_per_factor,
            "min_score": self.min_score,
            "action_clip": self.action_clip,
            "seed": self.seed,
            "fit_source": "training_split_only",
            "heldout_used_for_fitting": False,
            "heads": [head.metadata() for head in self.heads],
        }
