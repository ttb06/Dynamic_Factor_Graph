#!/usr/bin/env python3
"""Posterior-smoothed row/time routing for quantile-conditioned transport.

The existing quantile transport selects one hard target bin from the current
clean preview.  Near a bin boundary, small diffusion noise can therefore swap
the active operator abruptly.  This module keeps the train-fitted operators
but marginalizes the two nearest bins with a smooth, row-specific posterior.

``freeze_membership`` stores the first posterior and reuses it at later pulses.
Comparing frozen and recomputed membership separates row conditioning from
time-dependent graph reconfiguration without changing the action budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from batch_distribution_transport import BatchDistributionTransport


class PosteriorQuantileConditionalBatchDistributionTransport:
    """Blend the two nearest train-fitted quantile operators per row."""

    def __init__(
        self,
        train: pd.DataFrame,
        numeric_columns: list[str],
        edges: list[dict],
        condition_column: str,
        quantile_bins: int = 4,
        temperature: float = 0.75,
        freeze_membership: bool = False,
        scope: str = "all",
        mode: str = "copula",
        adaptive_top_k: int = 0,
        adaptive_min_gap: float = 0.0,
        min_group_rows: int = 64,
        preserve_tail_fraction: float = 0.0,
        sliced_directions: int = 32,
        seed: int = 20261019,
    ):
        if condition_column not in train:
            raise ValueError(f"condition column not found: {condition_column}")
        if temperature <= 0:
            raise ValueError("posterior temperature must be positive")
        self.condition_column = condition_column
        self.temperature = float(temperature)
        self.freeze_membership = bool(freeze_membership)
        self.global_operator = BatchDistributionTransport(
            train,
            numeric_columns,
            edges,
            scope=scope,
            mode=mode,
            adaptive_top_k=adaptive_top_k,
            adaptive_min_gap=adaptive_min_gap,
            preserve_tail_fraction=preserve_tail_fraction,
            sliced_directions=sliced_directions,
            seed=seed,
        )
        self.columns = self.global_operator.columns

        values = pd.to_numeric(train[condition_column], errors="coerce")
        self.fill_value = float(values.median())
        values_np = values.fillna(self.fill_value).to_numpy(float)
        requested = max(int(quantile_bins), 2)
        boundaries = np.unique(
            np.quantile(values_np, np.linspace(0.0, 1.0, requested + 1))
        )
        if len(boundaries) < 3:
            raise ValueError("numeric condition has fewer than two quantile bins")
        self.boundaries = boundaries
        self.bin_count = len(boundaries) - 1
        quantile_grid = (np.arange(self.bin_count, dtype=float) + 0.5) / self.bin_count
        self.centers = np.quantile(values_np, quantile_grid)
        global_scale = max(float(np.subtract(*np.percentile(values_np, [75, 25]))), 1e-6)
        if self.bin_count == 2:
            spacing = np.repeat(max(abs(self.centers[1] - self.centers[0]), 1e-6), 2)
        else:
            spacing = np.empty(self.bin_count, dtype=float)
            spacing[0] = self.centers[1] - self.centers[0]
            spacing[-1] = self.centers[-1] - self.centers[-2]
            spacing[1:-1] = 0.5 * (self.centers[2:] - self.centers[:-2])
            spacing = np.maximum(np.abs(spacing), 1e-3 * global_scale)
        self.spacing = spacing

        hard_labels = np.digitize(values_np, boundaries[1:-1], right=True)
        self.operators: dict[int, BatchDistributionTransport] = {}
        for label in np.unique(hard_labels):
            group = train.iloc[np.flatnonzero(hard_labels == label)]
            if len(group) < min_group_rows:
                continue
            self.operators[int(label)] = BatchDistributionTransport(
                group,
                numeric_columns,
                edges,
                scope=scope,
                mode=mode,
                adaptive_top_k=adaptive_top_k,
                adaptive_min_gap=adaptive_min_gap,
                preserve_tail_fraction=preserve_tail_fraction,
                sliced_directions=sliced_directions,
                seed=seed + int(label) + 1,
            )

        self._frozen_membership: np.ndarray | None = None
        self._previous_primary: np.ndarray | None = None
        self._call_count = 0

    def _membership(self, frame: pd.DataFrame) -> np.ndarray:
        if (
            self.freeze_membership
            and self._frozen_membership is not None
            and len(self._frozen_membership) == len(frame)
        ):
            return self._frozen_membership.copy()

        condition = pd.to_numeric(frame[self.condition_column], errors="coerce")
        values = condition.fillna(self.fill_value).to_numpy(float)
        distance = np.abs(values[:, None] - self.centers[None, :])
        distance = distance / self.spacing[None, :]
        nearest = np.argpartition(distance, kth=1, axis=1)[:, :2]
        selected_distance = np.take_along_axis(distance, nearest, axis=1)
        logits = -0.5 * np.square(selected_distance / self.temperature)
        logits -= logits.max(axis=1, keepdims=True)
        selected_weight = np.exp(logits)
        selected_weight /= selected_weight.sum(axis=1, keepdims=True)
        membership = np.zeros((len(frame), self.bin_count), dtype=float)
        rows = np.arange(len(frame))[:, None]
        membership[rows, nearest] = selected_weight
        if self.freeze_membership:
            self._frozen_membership = membership.copy()
        return membership

    def apply(
        self, frame: pd.DataFrame, strength: float
    ) -> tuple[pd.DataFrame, np.ndarray, dict]:
        membership = self._membership(frame)
        primary = membership.argmax(axis=1)
        switch_fraction = (
            float(np.mean(primary != self._previous_primary))
            if self._previous_primary is not None
            and len(self._previous_primary) == len(primary)
            else 0.0
        )
        self._previous_primary = primary.copy()
        self._call_count += 1

        current = frame[self.columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        accumulated_delta = np.zeros_like(current)
        diagnostic_mass: dict[str, float] = {}
        diagnostic_weight: dict[str, float] = {}
        groups_used = 0

        for label in range(self.bin_count):
            row_weight = membership[:, label]
            positions = np.flatnonzero(row_weight > 1e-6)
            if len(positions) < 2:
                continue
            operator = self.operators.get(label, self.global_operator)
            local_frame, _, diagnostics = operator.apply(
                frame.iloc[positions].copy(), strength
            )
            local_current = current[positions]
            local_updated = local_frame[self.columns].to_numpy(float)
            weight = row_weight[positions, None]
            accumulated_delta[positions] += weight * (local_updated - local_current)
            mass = float(row_weight[positions].sum())
            for key, metric in diagnostics.items():
                if np.isscalar(metric) and np.isfinite(float(metric)):
                    diagnostic_mass[key] = diagnostic_mass.get(key, 0.0) + mass * float(metric)
                    diagnostic_weight[key] = diagnostic_weight.get(key, 0.0) + mass
            groups_used += 1

        updated = current + accumulated_delta
        result = frame.copy()
        result.loc[:, self.columns] = updated
        tolerance = 8 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(current))
        changed = np.abs(accumulated_delta) > tolerance
        entropy = -np.sum(
            np.where(membership > 0, membership * np.log(np.maximum(membership, 1e-12)), 0.0),
            axis=1,
        ) / np.log(max(self.bin_count, 2))

        totals = {
            key: diagnostic_mass[key] / max(diagnostic_weight[key], 1e-12)
            for key in diagnostic_mass
        }
        totals.update(
            {
                "transport_columns": len(self.columns),
                "transport_cell_events": int(changed.sum()),
                "transport_row_events": int(changed.any(axis=1).sum()),
                "transport_condition_groups": groups_used,
                "transport_posterior_entropy_mean": float(entropy.mean()),
                "transport_posterior_entropy_p90": float(np.quantile(entropy, 0.90)),
                "transport_posterior_max_probability_mean": float(
                    membership.max(axis=1).mean()
                ),
                "transport_bin_switch_fraction": switch_fraction,
                "transport_membership_frozen": int(self.freeze_membership),
                "transport_condition_call_index": self._call_count,
                "transport_soft_temperature": self.temperature,
                "transport_mean_absolute_move": float(
                    np.mean(np.abs(accumulated_delta))
                ),
            }
        )
        return result, changed, totals
