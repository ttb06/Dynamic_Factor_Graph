#!/usr/bin/env python3
"""Adaptive-resolution posterior quantile transport.

The router fits q2, q4 and q8 posterior-smoothed conditional transports on
training data.  At every pulse it measures each row's fine-router posterior
confidence and forms a convex blend of the three candidate moves: uncertain
rows use coarse operators, intermediate rows use q4, and confident rows use
fine q8 operators.  Membership and resolution weights are recomputed at each
call unless ``freeze_membership`` is enabled.

Target preservation and tail reservation are applied *after* the resolution
blend.  This keeps those safeguards global rather than independently selecting
different protected rows inside each conditional group.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from batch_distribution_transport import stable_gaussian_ranks
from uncertainty_aware_quantile_transport import (
    UncertaintyAwarePosteriorQuantileTransport,
)


class AdaptiveResolutionQuantileTransport:
    """Blend q2/q4/q8 conditional transports using row-wise uncertainty.

    Parameters mirror the existing posterior transport where possible.  The
    confidence thresholds operate on the maximum posterior probability of the
    q8 router.  With the default posterior temperature, interior-bin confidence
    is usually in roughly ``[0.5, 0.71]``; consequently conservative defaults
    of 0.52 and 0.68 use all three resolutions without hard switching.

    ``preserve_target`` protects ``condition_column`` when it is among the
    transported columns.  ``preserve_condition_column`` is accepted as a
    compatibility alias.  Likewise, ``preserve_tail_fraction`` aliases the
    explicitly named ``global_tail_reserve_fraction``.
    """

    RESOLUTIONS = (2, 4, 8)

    def __init__(
        self,
        train: pd.DataFrame,
        numeric_columns: list[str],
        edges: list[dict],
        condition_column: str,
        temperature: float = 0.75,
        confidence_floor: float = 0.52,
        confidence_full: float = 0.68,
        freeze_membership: bool = False,
        preserve_target: bool = False,
        global_tail_reserve_fraction: float = 0.0,
        scope: str = "all",
        mode: str = "copula",
        adaptive_top_k: int = 0,
        adaptive_min_gap: float = 0.0,
        min_group_rows: int = 64,
        sliced_directions: int = 32,
        seed: int = 20261019,
        *,
        preserve_condition_column: bool | None = None,
        preserve_tail_fraction: float | None = None,
    ):
        if condition_column not in train:
            raise ValueError(f"condition column not found: {condition_column}")
        if temperature <= 0.0:
            raise ValueError("posterior temperature must be positive")
        if not 0.0 <= confidence_floor < confidence_full <= 1.0:
            raise ValueError(
                "confidence thresholds must satisfy "
                "0 <= confidence_floor < confidence_full <= 1"
            )
        if min_group_rows < 2:
            raise ValueError("min_group_rows must be at least 2")

        if preserve_condition_column is not None:
            preserve_target = bool(preserve_condition_column)
        if preserve_tail_fraction is not None:
            if (
                global_tail_reserve_fraction != 0.0
                and not np.isclose(
                    global_tail_reserve_fraction,
                    preserve_tail_fraction,
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    "global_tail_reserve_fraction and preserve_tail_fraction "
                    "specify different values"
                )
            global_tail_reserve_fraction = float(preserve_tail_fraction)
        if not 0.0 <= global_tail_reserve_fraction < 1.0:
            raise ValueError(
                "global_tail_reserve_fraction must lie in [0, 1)"
            )

        self.condition_column = condition_column
        self.temperature = float(temperature)
        self.confidence_floor = float(confidence_floor)
        self.confidence_full = float(confidence_full)
        self.freeze_membership = bool(freeze_membership)
        self.preserve_target = bool(preserve_target)
        self.preserve_condition_column = self.preserve_target
        self.global_tail_reserve_fraction = float(
            global_tail_reserve_fraction
        )
        self.coverage_reserve_fraction = self.global_tail_reserve_fraction
        self.seed = int(seed)

        self.routers: dict[int, UncertaintyAwarePosteriorQuantileTransport] = {}
        for ordinal, resolution in enumerate(self.RESOLUTIONS):
            router = UncertaintyAwarePosteriorQuantileTransport(
                train,
                numeric_columns,
                edges,
                condition_column=condition_column,
                quantile_bins=resolution,
                temperature=self.temperature,
                freeze_membership=self.freeze_membership,
                scope=scope,
                mode=mode,
                adaptive_top_k=adaptive_top_k,
                adaptive_min_gap=adaptive_min_gap,
                min_group_rows=min_group_rows,
                preserve_tail_fraction=0.0,
                sliced_directions=sliced_directions,
                seed=self.seed + 1009 * ordinal,
                confidence_floor=0.0,
                confidence_full=1.0,
                uncertainty_fallback="none",
                fallback_scale=1.0,
                membership_hysteresis=0.0,
                switch_margin=0.0,
                preserve_condition_column=False,
            )
            self.routers[resolution] = router

        first = self.routers[self.RESOLUTIONS[0]]
        self.columns = list(first.columns)
        if any(list(router.columns) != self.columns for router in self.routers.values()):
            raise ValueError("q2/q4/q8 routers selected inconsistent columns")
        # Every resolution uses the same full-training fallback and robust
        # coordinate system.  Sharing it also avoids keeping three identical
        # global reference matrices alive.
        self.global_operator = first.global_operator
        for router in self.routers.values():
            router.global_operator = self.global_operator

        self._call_count = 0
        self._frozen_resolution_weights: np.ndarray | None = None
        self._frozen_fine_membership: np.ndarray | None = None
        self._frozen_frame_index: pd.Index | None = None
        self._previous_resolution_weights: np.ndarray | None = None
        self._previous_primary_resolution: np.ndarray | None = None
        self._previous_frame_index: pd.Index | None = None

    @staticmethod
    def _finite_scalar(value: object) -> bool:
        return bool(np.isscalar(value) and np.isfinite(float(value)))

    @staticmethod
    def _posterior_membership(
        router: UncertaintyAwarePosteriorQuantileTransport,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """Compute the router posterior without mutating its temporal state."""
        condition = pd.to_numeric(
            frame[router.condition_column], errors="coerce"
        )
        values = condition.fillna(router.fill_value).to_numpy(float)
        distance = np.abs(values[:, None] - router.centers[None, :])
        distance = distance / router.spacing[None, :]
        nearest = np.argpartition(distance, kth=1, axis=1)[:, :2]
        selected_distance = np.take_along_axis(distance, nearest, axis=1)
        logits = -0.5 * np.square(
            selected_distance / router.temperature
        )
        logits -= logits.max(axis=1, keepdims=True)
        selected_weight = np.exp(logits)
        selected_weight /= np.maximum(
            selected_weight.sum(axis=1, keepdims=True), 1e-12
        )
        membership = np.zeros(
            (len(frame), router.bin_count), dtype=np.float64
        )
        membership[np.arange(len(frame))[:, None], nearest] = selected_weight
        return membership

    def _reset_frozen_state_for_new_rows(self, frame: pd.DataFrame) -> None:
        if not self.freeze_membership:
            return
        if (
            self._frozen_frame_index is not None
            and not frame.index.equals(self._frozen_frame_index)
        ):
            self._frozen_resolution_weights = None
            self._frozen_fine_membership = None
            for router in self.routers.values():
                router._frozen_membership = None
        if self._frozen_frame_index is None or self._frozen_resolution_weights is None:
            self._frozen_frame_index = frame.index.copy()

    def _resolution_weights(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        self._reset_frozen_state_for_new_rows(frame)
        if (
            self.freeze_membership
            and self._frozen_resolution_weights is not None
            and self._frozen_fine_membership is not None
            and len(self._frozen_resolution_weights) == len(frame)
        ):
            return (
                self._frozen_resolution_weights.copy(),
                self._frozen_fine_membership.copy(),
            )

        fine_membership = self._posterior_membership(
            self.routers[self.RESOLUTIONS[-1]], frame
        )
        confidence = fine_membership.max(axis=1)
        progress = np.clip(
            (confidence - self.confidence_floor)
            / max(self.confidence_full - self.confidence_floor, 1e-12),
            0.0,
            1.0,
        ) * 2.0
        weights = np.column_stack(
            (
                np.clip(1.0 - progress, 0.0, 1.0),
                np.clip(1.0 - np.abs(progress - 1.0), 0.0, 1.0),
                np.clip(progress - 1.0, 0.0, 1.0),
            )
        )
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

        if self.freeze_membership:
            self._frozen_resolution_weights = weights.copy()
            self._frozen_fine_membership = fine_membership.copy()
            self._frozen_frame_index = frame.index.copy()
        return weights, fine_membership

    def _current_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        current_frame = frame[self.columns].apply(
            pd.to_numeric, errors="coerce"
        )
        fill = pd.Series(
            dict(zip(self.columns, self.global_operator.medians))
        )
        return current_frame.fillna(fill).to_numpy(np.float64)

    def apply(
        self, frame: pd.DataFrame, strength: float
    ) -> tuple[pd.DataFrame, np.ndarray, dict]:
        """Apply one adaptive-resolution pulse.

        Returns ``(frame, changed, diagnostics)``.  ``changed`` has shape
        ``(len(frame), len(columns))`` and every diagnostic value is scalar.
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError("transport strength must lie in [0, 1]")
        if len(frame) == 0:
            raise ValueError("adaptive-resolution transport requires a non-empty frame")
        missing = [
            column
            for column in dict.fromkeys([self.condition_column, *self.columns])
            if column not in frame
        ]
        if missing:
            raise ValueError(f"frame is missing required columns: {missing}")

        weights, fine_membership = self._resolution_weights(frame)
        current = self._current_matrix(frame)
        resolution_deltas: list[np.ndarray] = []
        router_diagnostics: dict[int, Mapping[str, object]] = {}

        for resolution in self.RESOLUTIONS:
            routed_frame, _, diagnostics = self.routers[resolution].apply(
                frame.copy(), strength
            )
            routed = routed_frame[self.columns].apply(
                pd.to_numeric, errors="coerce"
            ).fillna(
                pd.Series(
                    dict(zip(self.columns, self.global_operator.medians))
                )
            ).to_numpy(np.float64)
            resolution_deltas.append(routed - current)
            router_diagnostics[resolution] = diagnostics

        delta_stack = np.stack(resolution_deltas, axis=0)
        accumulated_delta = np.einsum(
            "nr,rnd->nd", weights, delta_stack, optimize=True
        )
        pre_preserve_delta = accumulated_delta.copy()

        target_move_before_preserve = 0.0
        target_preserved = False
        if self.preserve_target and self.condition_column in self.columns:
            target_index = self.columns.index(self.condition_column)
            target_move_before_preserve = float(
                np.mean(np.abs(accumulated_delta[:, target_index]))
            )
            accumulated_delta[:, target_index] = 0.0
            target_preserved = True

        pre_reserve_delta = accumulated_delta.copy()
        protected_rows = np.zeros(len(current), dtype=bool)
        if self.global_tail_reserve_fraction > 0.0 and len(current) > 2:
            gaussian_rarity = np.mean(
                stable_gaussian_ranks(current) ** 2, axis=1
            )
            threshold = np.quantile(
                gaussian_rarity,
                1.0 - self.global_tail_reserve_fraction,
            )
            protected_rows = gaussian_rarity >= threshold
            accumulated_delta[protected_rows] = 0.0

        updated = current + accumulated_delta
        result = frame.copy()
        for column_index, column in enumerate(self.columns):
            values = updated[:, column_index]
            if pd.api.types.is_float_dtype(result[column].dtype):
                values = values.astype(result[column].dtype, copy=False)
            result[column] = values
        if target_preserved:
            # Preserve the exact incoming values and dtype, including NaNs;
            # zeroing the numeric delta alone would replace NaNs with the
            # train median used for stable transport arithmetic.
            result[self.condition_column] = frame[
                self.condition_column
            ].copy()

        tolerance = 8 * np.finfo(np.float32).eps * np.maximum(
            1.0, np.abs(current)
        )
        changed = np.abs(accumulated_delta) > tolerance

        confidence = fine_membership.max(axis=1)
        posterior_entropy = -np.sum(
            np.where(
                fine_membership > 0.0,
                fine_membership
                * np.log(np.maximum(fine_membership, 1e-12)),
                0.0,
            ),
            axis=1,
        ) / np.log(2.0)
        blend_entropy = -np.sum(
            np.where(
                weights > 0.0,
                weights * np.log(np.maximum(weights, 1e-12)),
                0.0,
            ),
            axis=1,
        ) / np.log(float(len(self.RESOLUTIONS)))
        resolution_values = np.asarray(self.RESOLUTIONS, dtype=np.float64)
        expected_resolution = weights @ resolution_values
        primary_resolution_index = weights.argmax(axis=1)

        same_rows = (
            self._previous_frame_index is not None
            and frame.index.equals(self._previous_frame_index)
            and self._previous_resolution_weights is not None
            and len(self._previous_resolution_weights) == len(frame)
        )
        if same_rows:
            resolution_weight_l1_drift = float(
                np.mean(
                    np.abs(
                        weights - self._previous_resolution_weights
                    ).sum(axis=1)
                )
            )
            resolution_switch_fraction = float(
                np.mean(
                    primary_resolution_index
                    != self._previous_primary_resolution
                )
            )
        else:
            resolution_weight_l1_drift = 0.0
            resolution_switch_fraction = 0.0
        self._previous_resolution_weights = weights.copy()
        self._previous_primary_resolution = primary_resolution_index.copy()
        self._previous_frame_index = frame.index.copy()
        self._call_count += 1

        robust_scale = self.global_operator.robust_scale[None, :]
        blend_center = np.einsum(
            "nr,rnd->nd", weights, delta_stack, optimize=True
        )
        weighted_variance = np.einsum(
            "nr,rnd->nd",
            weights,
            np.square(delta_stack - blend_center[None, :, :]),
            optimize=True,
        )
        standardized_disagreement = np.sqrt(
            np.maximum(weighted_variance, 0.0)
        ) / robust_scale

        fine_diagnostics = router_diagnostics[self.RESOLUTIONS[-1]]
        q4_diagnostics = router_diagnostics[4]
        total_condition_groups = sum(
            int(router_diagnostics[resolution].get(
                "transport_condition_groups", 0
            ))
            for resolution in self.RESOLUTIONS
        )
        diagnostics: dict[str, float | int] = {
            "transport_columns": len(self.columns),
            "transport_cell_events": int(changed.sum()),
            "transport_row_events": int(changed.any(axis=1).sum()),
            "transport_condition_groups": total_condition_groups,
            "transport_condition_call_index": self._call_count,
            "transport_posterior_entropy_mean": float(
                posterior_entropy.mean()
            ),
            "transport_posterior_entropy_p90": float(
                np.quantile(posterior_entropy, 0.90)
            ),
            "transport_posterior_max_probability_mean": float(
                confidence.mean()
            ),
            "transport_bin_switch_fraction": float(
                fine_diagnostics.get("transport_bin_switch_fraction", 0.0)
            ),
            "transport_membership_l1_drift": float(
                fine_diagnostics.get("transport_membership_l1_drift", 0.0)
            ),
            "transport_membership_frozen": int(self.freeze_membership),
            "transport_soft_temperature": self.temperature,
            "transport_confidence_floor": self.confidence_floor,
            "transport_confidence_full": self.confidence_full,
            "transport_uncertain_row_events": int(
                np.count_nonzero(confidence < self.confidence_full)
            ),
            "transport_local_mean_absolute_move": float(
                np.mean(np.abs(pre_reserve_delta))
            ),
            "transport_global_mean_absolute_move": 0.0,
            "transport_adaptive_resolution_router_count": len(
                self.RESOLUTIONS
            ),
            "transport_adaptive_resolution_membership_frozen": int(
                self.freeze_membership
            ),
            "transport_adaptive_resolution_confidence_floor": (
                self.confidence_floor
            ),
            "transport_adaptive_resolution_confidence_full": (
                self.confidence_full
            ),
            "transport_adaptive_resolution_confidence_mean": float(
                confidence.mean()
            ),
            "transport_adaptive_resolution_confidence_p10": float(
                np.quantile(confidence, 0.10)
            ),
            "transport_adaptive_resolution_confidence_p90": float(
                np.quantile(confidence, 0.90)
            ),
            "transport_adaptive_resolution_posterior_entropy_mean": float(
                posterior_entropy.mean()
            ),
            "transport_adaptive_resolution_posterior_entropy_p90": float(
                np.quantile(posterior_entropy, 0.90)
            ),
            "transport_adaptive_resolution_blend_entropy_mean": float(
                blend_entropy.mean()
            ),
            "transport_adaptive_resolution_blend_entropy_p90": float(
                np.quantile(blend_entropy, 0.90)
            ),
            "transport_adaptive_resolution_expected_bins_mean": float(
                expected_resolution.mean()
            ),
            "transport_adaptive_resolution_expected_bins_p10": float(
                np.quantile(expected_resolution, 0.10)
            ),
            "transport_adaptive_resolution_expected_bins_p90": float(
                np.quantile(expected_resolution, 0.90)
            ),
            "transport_resolution_mean": float(
                expected_resolution.mean()
            ),
            "transport_resolution_switch_fraction": (
                resolution_switch_fraction
            ),
            "transport_coarse_row_fraction": float(
                np.mean(primary_resolution_index == 0)
            ),
            "transport_fine_row_fraction": float(
                np.mean(primary_resolution_index == 2)
            ),
            "transport_effective_bin_count_mean": float(
                expected_resolution.mean()
            ),
            "transport_adaptive_resolution_switch_fraction": (
                resolution_switch_fraction
            ),
            "transport_adaptive_resolution_weight_l1_drift": (
                resolution_weight_l1_drift
            ),
            "transport_adaptive_resolution_delta_disagreement_mean": float(
                standardized_disagreement.mean()
            ),
            "transport_adaptive_resolution_delta_disagreement_p90": float(
                np.quantile(
                    standardized_disagreement.mean(axis=1), 0.90
                )
            ),
            "transport_condition_column_preserved": int(target_preserved),
            "transport_target_mean_absolute_move_before_preserve": (
                target_move_before_preserve
            ),
            "transport_global_tail_reserve_requested_fraction": (
                self.global_tail_reserve_fraction
            ),
            "transport_tail_rows_preserved": int(protected_rows.sum()),
            "transport_tail_fraction_preserved": float(
                protected_rows.mean()
            ),
            "transport_mean_absolute_move": float(
                np.mean(np.abs(accumulated_delta))
            ),
            "transport_pre_reserve_mean_absolute_move": float(
                np.mean(np.abs(pre_reserve_delta))
            ),
            "transport_pre_target_preserve_mean_absolute_move": float(
                np.mean(np.abs(pre_preserve_delta))
            ),
            "transport_standardized_move_p90": float(
                np.quantile(
                    np.mean(
                        np.abs(accumulated_delta) / robust_scale,
                        axis=1,
                    ),
                    0.90,
                )
            ),
        }

        # Preserve the canonical q4 bin diagnostics consumed by the current
        # driver while the full q2/q4/q8 diagnostics remain namespaced below.
        for label in range(4):
            for suffix in ("posterior_mass", "primary_fraction"):
                key = f"transport_bin_{label}_{suffix}"
                diagnostics[key] = float(q4_diagnostics.get(key, 0.0))

        for resolution_index, resolution in enumerate(self.RESOLUTIONS):
            prefix = f"transport_adaptive_resolution_q{resolution}"
            diagnostics[f"{prefix}_weight_mean"] = float(
                weights[:, resolution_index].mean()
            )
            diagnostics[f"{prefix}_primary_fraction"] = float(
                np.mean(primary_resolution_index == resolution_index)
            )
            diagnostics[f"{prefix}_mean_absolute_move"] = float(
                np.mean(np.abs(delta_stack[resolution_index]))
            )
            diagnostics[f"{prefix}_actual_bins"] = int(
                self.routers[resolution].bin_count
            )
            diagnostics[f"{prefix}_fitted_local_operators"] = len(
                self.routers[resolution].operators
            )
            for key, value in router_diagnostics[resolution].items():
                if self._finite_scalar(value):
                    suffix = key.removeprefix("transport_")
                    diagnostics[f"{prefix}_{suffix}"] = float(value)

        return result, changed, diagnostics


# Longer explicit name for call sites that distinguish this from hard-bin
# adaptive transports.  Both names intentionally refer to the same class.
AdaptiveResolutionPosteriorQuantileTransport = (
    AdaptiveResolutionQuantileTransport
)

