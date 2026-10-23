#!/usr/bin/env python3
"""Uncertainty-aware posterior routing for quantile-conditioned transport.

This extends posterior-smoothed routing with two safeguards that can be
ablated independently: temporal membership hysteresis and an uncertainty
fallback.  The latter either blends uncertain rows toward a global operator
or abstains, while confident rows retain their conditional operator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from batch_distribution_transport import stable_gaussian_ranks
from posterior_quantile_transport import (
    PosteriorQuantileConditionalBatchDistributionTransport,
)


class UncertaintyAwarePosteriorQuantileTransport(
    PosteriorQuantileConditionalBatchDistributionTransport
):
    """Posterior router with row-wise confidence and temporal hysteresis."""

    def __init__(
        self,
        *args,
        confidence_floor: float = 0.0,
        confidence_full: float = 1.0,
        uncertainty_fallback: str = "none",
        fallback_scale: float = 1.0,
        membership_hysteresis: float = 0.0,
        switch_margin: float = 0.0,
        preserve_condition_column: bool = False,
        **kwargs,
    ):
        coverage_reserve_fraction = float(
            kwargs.pop("preserve_tail_fraction", 0.0)
        )
        super().__init__(*args, preserve_tail_fraction=0.0, **kwargs)
        if not 0.0 <= confidence_floor < 1.0:
            raise ValueError("confidence_floor must be in [0, 1)")
        if not confidence_floor < confidence_full <= 1.0:
            raise ValueError("confidence_full must be in (confidence_floor, 1]")
        if uncertainty_fallback not in {"none", "global", "abstain"}:
            raise ValueError(
                "uncertainty_fallback must be one of: none, global, abstain"
            )
        if not 0.0 <= membership_hysteresis < 1.0:
            raise ValueError("membership_hysteresis must be in [0, 1)")
        if not 0.0 <= fallback_scale <= 1.0:
            raise ValueError("fallback_scale must be in [0, 1]")
        if not 0.0 <= switch_margin <= 1.0:
            raise ValueError("switch_margin must be in [0, 1]")
        if not 0.0 <= coverage_reserve_fraction < 1.0:
            raise ValueError("preserve_tail_fraction must be in [0, 1)")
        self.confidence_floor = float(confidence_floor)
        self.confidence_full = float(confidence_full)
        self.uncertainty_fallback = uncertainty_fallback
        self.fallback_scale = float(fallback_scale)
        self.membership_hysteresis = float(membership_hysteresis)
        self.switch_margin = float(switch_margin)
        self.preserve_condition_column = bool(preserve_condition_column)
        self.coverage_reserve_fraction = coverage_reserve_fraction
        self._previous_membership: np.ndarray | None = None
        self._previous_frame_index: pd.Index | None = None
        self._last_membership_l1_drift = 0.0
        self._last_switch_candidate_fraction = 0.0
        self._last_switch_held_fraction = 0.0

    def _membership(self, frame: pd.DataFrame) -> np.ndarray:
        membership = super()._membership(frame)
        previous = self._previous_membership
        same_rows = (
            previous is not None
            and len(previous) == len(membership)
            and self._previous_frame_index is not None
            and frame.index.equals(self._previous_frame_index)
        )
        self._last_switch_candidate_fraction = 0.0
        self._last_switch_held_fraction = 0.0
        if same_rows:
            self._last_membership_l1_drift = float(
                np.mean(np.abs(membership - previous).sum(axis=1))
            )
            if not self.freeze_membership and self.switch_margin > 0.0:
                candidate = membership.argmax(axis=1)
                old = previous.argmax(axis=1)
                switched = candidate != old
                rows = np.arange(len(membership))
                advantage = (
                    membership[rows, candidate] - membership[rows, old]
                )
                held = switched & (advantage < self.switch_margin)
                membership[held] = previous[held]
                self._last_switch_candidate_fraction = float(switched.mean())
                self._last_switch_held_fraction = float(held.mean())
            if not self.freeze_membership and self.membership_hysteresis > 0.0:
                membership = (
                    self.membership_hysteresis * previous
                    + (1.0 - self.membership_hysteresis) * membership
                )
                membership /= np.maximum(
                    membership.sum(axis=1, keepdims=True), 1e-12
                )
        else:
            self._last_membership_l1_drift = 0.0
        self._previous_membership = membership.copy()
        self._previous_frame_index = frame.index.copy()
        return membership

    def _confidence_gate(self, membership: np.ndarray) -> np.ndarray:
        max_probability = membership.max(axis=1)
        if self.uncertainty_fallback == "none":
            return np.ones(len(membership), dtype=float)
        return np.clip(
            (max_probability - self.confidence_floor)
            / max(self.confidence_full - self.confidence_floor, 1e-12),
            0.0,
            1.0,
        )

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

        current_frame = frame[self.columns].apply(
            pd.to_numeric, errors="coerce"
        )
        current = current_frame.fillna(
            pd.Series(dict(zip(self.columns, self.global_operator.medians)))
        ).to_numpy(float)
        conditional_delta = np.zeros_like(current)
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
            conditional_delta[positions] += weight * (
                local_updated - local_current
            )
            mass = float(row_weight[positions].sum())
            for key, metric in diagnostics.items():
                if np.isscalar(metric) and np.isfinite(float(metric)):
                    diagnostic_mass[key] = (
                        diagnostic_mass.get(key, 0.0) + mass * float(metric)
                    )
                    diagnostic_weight[key] = (
                        diagnostic_weight.get(key, 0.0) + mass
                    )
            groups_used += 1

        gate = self._confidence_gate(membership)
        global_delta = np.zeros_like(current)
        if self.uncertainty_fallback == "global":
            global_frame, _, global_diagnostics = self.global_operator.apply(
                frame.copy(), strength
            )
            global_delta = (
                global_frame[self.columns].to_numpy(float) - current
            )
            accumulated_delta = (
                gate[:, None] * conditional_delta
                + (
                    (1.0 - gate[:, None])
                    * self.fallback_scale
                    * global_delta
                )
            )
        elif self.uncertainty_fallback == "abstain":
            global_diagnostics = {}
            accumulated_delta = gate[:, None] * conditional_delta
        else:
            global_diagnostics = {}
            accumulated_delta = conditional_delta

        if (
            self.preserve_condition_column
            and self.condition_column in self.columns
        ):
            condition_index = self.columns.index(self.condition_column)
            accumulated_delta[:, condition_index] = 0.0

        pre_reserve_delta = accumulated_delta.copy()
        protected_rows = np.zeros(len(current), dtype=bool)
        if self.coverage_reserve_fraction > 0.0 and len(current) > 2:
            gaussian_rarity = np.mean(
                stable_gaussian_ranks(current) ** 2, axis=1
            )
            threshold = np.quantile(
                gaussian_rarity, 1.0 - self.coverage_reserve_fraction
            )
            protected_rows = gaussian_rarity >= threshold
            accumulated_delta[protected_rows] = 0.0

        updated = current + accumulated_delta
        result = frame.copy()
        for index, column in enumerate(self.columns):
            values = updated[:, index]
            if pd.api.types.is_float_dtype(result[column].dtype):
                values = values.astype(result[column].dtype, copy=False)
            result[column] = values

        tolerance = 8 * np.finfo(np.float32).eps * np.maximum(
            1.0, np.abs(current)
        )
        changed = np.abs(accumulated_delta) > tolerance
        entropy = -np.sum(
            np.where(
                membership > 0,
                membership * np.log(np.maximum(membership, 1e-12)),
                0.0,
            ),
            axis=1,
        ) / np.log(2.0)

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
                "transport_posterior_entropy_p90": float(
                    np.quantile(entropy, 0.90)
                ),
                "transport_posterior_max_probability_mean": float(
                    membership.max(axis=1).mean()
                ),
                "transport_bin_switch_fraction": switch_fraction,
                "transport_membership_l1_drift": (
                    self._last_membership_l1_drift
                ),
                "transport_membership_hysteresis": (
                    self.membership_hysteresis
                ),
                "transport_switch_margin": self.switch_margin,
                "transport_switch_candidate_fraction": (
                    self._last_switch_candidate_fraction
                ),
                "transport_switch_held_fraction": (
                    self._last_switch_held_fraction
                ),
                "transport_confidence_floor": self.confidence_floor,
                "transport_confidence_full": self.confidence_full,
                "transport_fallback_scale": self.fallback_scale,
                "transport_confidence_gate_mean": float(gate.mean()),
                "transport_confidence_gate_p10": float(
                    np.quantile(gate, 0.10)
                ),
                "transport_uncertain_row_events": int(
                    np.count_nonzero(gate < 1.0 - 1e-9)
                ),
                "transport_uncertainty_fallback_global": int(
                    self.uncertainty_fallback == "global"
                ),
                "transport_uncertainty_fallback_abstain": int(
                    self.uncertainty_fallback == "abstain"
                ),
                "transport_condition_column_preserved": int(
                    self.preserve_condition_column
                ),
                "transport_tail_rows_preserved": int(protected_rows.sum()),
                "transport_tail_fraction_preserved": float(
                    protected_rows.mean()
                ),
                "transport_membership_frozen": int(self.freeze_membership),
                "transport_condition_call_index": self._call_count,
                "transport_soft_temperature": self.temperature,
                "transport_mean_absolute_move": float(
                    np.mean(np.abs(accumulated_delta))
                ),
                "transport_local_mean_absolute_move": float(
                    np.mean(np.abs(conditional_delta))
                ),
                "transport_global_mean_absolute_move": float(
                    np.mean(np.abs(global_delta))
                ),
                "transport_pre_reserve_mean_absolute_move": float(
                    np.mean(np.abs(pre_reserve_delta))
                ),
                "transport_standardized_move_p90": float(
                    np.quantile(
                        np.mean(
                            np.abs(accumulated_delta)
                            / self.global_operator.robust_scale[None, :],
                            axis=1,
                        ),
                        0.90,
                    )
                ),
            }
        )
        for label in range(self.bin_count):
            totals[f"transport_bin_{label}_posterior_mass"] = float(
                membership[:, label].mean()
            )
            totals[f"transport_bin_{label}_primary_fraction"] = float(
                np.mean(primary == label)
            )
        for key in (
            "transport_correlation_error_before",
            "transport_correlation_error_after",
            "transport_projection_error_before",
            "transport_projection_error_after",
        ):
            metric = global_diagnostics.get(key)
            if metric is not None and np.isscalar(metric):
                totals[f"transport_global_{key.removeprefix('transport_')}"] = (
                    float(metric)
                )
        return result, changed, totals
