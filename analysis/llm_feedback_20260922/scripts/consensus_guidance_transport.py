#!/usr/bin/env python3
"""Action-space consensus between a locked transport and a factor witness.

The locked reference action is retained exactly.  A cross-fitted factorized
witness contributes only a bounded component that is either coordinate-wise
aligned with the reference or projected out of the reference-conflicting
row-wise subspace (PCGrad-style gradient surgery).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


VALID_MODES = {"aligned", "pcgrad"}


def _scale(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    location = float(finite.median()) if len(finite) else 0.0
    q25, q75 = np.quantile(finite, [0.25, 0.75]) if len(finite) else (0.0, 1.0)
    spread = float(q75 - q25)
    if spread <= 1e-8:
        spread = float(finite.std()) if len(finite) else 1.0
    return location, max(spread, 1e-6)


class ConsensusGuidanceTransport:
    """Fuse a reference operator with a sparse witness under a norm budget."""

    def __init__(
        self,
        train: pd.DataFrame,
        reference_operator: Any,
        witness_operator: Any,
        *,
        mode: str = "pcgrad",
        witness_ratio: float = 0.25,
        target_column: str | None = None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"unsupported consensus mode: {mode}")
        if not (0.0 < witness_ratio <= 1.0):
            raise ValueError("witness ratio must be in (0,1]")
        self.mode = mode
        self.witness_ratio = float(witness_ratio)
        self.target_column = target_column
        self.reference = reference_operator
        self.witness = witness_operator
        self.columns = list(reference_operator.columns)
        self.fusion_columns = [
            column for column in witness_operator.columns
            if column in self.columns and column != target_column
        ]
        if not self.fusion_columns:
            raise ValueError("consensus has no non-target fusion columns")
        self.fill: dict[str, float] = {}
        self.scale: dict[str, float] = {}
        for column in self.columns:
            location, spread = _scale(train[column])
            self.fill[column] = location
            self.scale[column] = spread

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.columns].apply(pd.to_numeric, errors="coerce")
        for column in self.columns:
            values[column] = values[column].fillna(self.fill[column])
        return values.to_numpy(np.float64)

    def apply(self, frame: pd.DataFrame, strength: float) -> tuple[pd.DataFrame, np.ndarray, dict]:
        reference_frame, reference_changed, reference_diagnostics = self.reference.apply(
            frame, strength
        )
        witness_frame, witness_changed, witness_diagnostics = self.witness.apply(
            frame, strength
        )
        current = self._matrix(frame)
        reference_values = self._matrix(reference_frame)
        witness_values = self._matrix(witness_frame)
        reference_delta = reference_values - current
        witness_delta = witness_values - current

        fusion_positions = np.array(
            [self.columns.index(column) for column in self.fusion_columns], dtype=int
        )
        spread = np.array([self.scale[column] for column in self.fusion_columns])
        reference_z = reference_delta[:, fusion_positions] / spread[None, :]
        witness_z = witness_delta[:, fusion_positions] / spread[None, :]

        reference_norm = np.linalg.norm(reference_z, axis=1)
        witness_norm = np.linalg.norm(witness_z, axis=1)
        dot_before = np.sum(reference_z * witness_z, axis=1)
        active_pair = (reference_norm > 1e-12) & (witness_norm > 1e-12)
        conflict = active_pair & (dot_before < 0.0)

        if self.mode == "aligned":
            same_sign = reference_z * witness_z > 0.0
            filtered = np.where(same_sign, witness_z, 0.0)
        else:
            filtered = witness_z.copy()
            denominator = np.square(reference_norm)
            rows = np.flatnonzero(conflict & (denominator > 1e-12))
            if len(rows):
                coefficient = dot_before[rows] / denominator[rows]
                filtered[rows] -= coefficient[:, None] * reference_z[rows]
            filtered[reference_norm <= 1e-12] = 0.0

        filtered_norm = np.linalg.norm(filtered, axis=1)
        budget = self.witness_ratio * reference_norm
        shrink = np.ones(len(frame), dtype=np.float64)
        over = filtered_norm > budget
        shrink[over] = budget[over] / np.maximum(filtered_norm[over], 1e-12)
        filtered *= shrink[:, None]
        extra_delta = filtered * spread[None, :]

        combined_delta = reference_delta.copy()
        combined_delta[:, fusion_positions] += extra_delta
        combined_values = current + combined_delta
        result = reference_frame.copy()
        result.loc[:, self.columns] = combined_values
        tolerance = 8 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(current))
        changed = np.abs(combined_delta) > tolerance
        extra_changed = np.abs(extra_delta) > (
            8 * np.finfo(np.float32).eps
            * np.maximum(1.0, np.abs(current[:, fusion_positions]))
        )

        filtered_norm_after = np.linalg.norm(filtered, axis=1)
        dot_after = np.sum(reference_z * filtered, axis=1)
        cosine_before = np.zeros(len(frame), dtype=np.float64)
        cosine_after = np.zeros(len(frame), dtype=np.float64)
        cosine_before[active_pair] = dot_before[active_pair] / (
            reference_norm[active_pair] * witness_norm[active_pair]
        )
        after_active = (reference_norm > 1e-12) & (filtered_norm_after > 1e-12)
        cosine_after[after_active] = dot_after[after_active] / (
            reference_norm[after_active] * filtered_norm_after[after_active]
        )
        target_followed_reference = 1
        if self.target_column and self.target_column in frame:
            target_followed_reference = int(
                result[self.target_column].reset_index(drop=True).equals(
                    reference_frame[self.target_column].reset_index(drop=True)
                )
            )

        diagnostics: dict[str, Any] = {
            "transport_cell_events": int(changed.sum()),
            "transport_row_events": int(changed.any(axis=1).sum()),
            "transport_mean_absolute_move": float(np.abs(combined_delta).mean()),
            "consensus_mode_aligned": int(self.mode == "aligned"),
            "consensus_mode_pcgrad": int(self.mode == "pcgrad"),
            "consensus_witness_ratio": self.witness_ratio,
            "consensus_reference_cell_events": int(reference_changed.sum()),
            "consensus_witness_cell_events": int(witness_changed.sum()),
            "consensus_extra_cell_events": int(extra_changed.sum()),
            "consensus_extra_row_events": int(extra_changed.any(axis=1).sum()),
            "consensus_conflict_rows": int(conflict.sum()),
            "consensus_conflict_fraction": float(conflict.mean()),
            "consensus_cosine_before_mean": float(cosine_before[active_pair].mean() if active_pair.any() else 0.0),
            "consensus_cosine_after_mean": float(cosine_after[after_active].mean() if after_active.any() else 0.0),
            "consensus_extra_to_reference_norm_mean": float(
                np.mean(filtered_norm_after[reference_norm > 1e-12] / reference_norm[reference_norm > 1e-12])
                if np.any(reference_norm > 1e-12) else 0.0
            ),
            "consensus_target_followed_reference": target_followed_reference,
            "consensus_reference_mean_absolute_move": float(np.abs(reference_delta).mean()),
            "consensus_extra_mean_absolute_move": float(np.abs(extra_delta).mean()),
            "consensus_witness_selected_cells": int(
                witness_diagnostics.get("residual_witness_selected_cells", 0)
            ),
            "consensus_witness_selected_rows": int(
                witness_diagnostics.get("residual_witness_selected_rows", 0)
            ),
        }
        for key, value in reference_diagnostics.items():
            if np.isscalar(value):
                diagnostics[f"consensus_reference_{key}"] = value
        for key, value in witness_diagnostics.items():
            if np.isscalar(value):
                diagnostics[f"consensus_witness_{key}"] = value
        return result, changed, diagnostics

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "witness_ratio": self.witness_ratio,
            "columns": list(self.columns),
            "fusion_columns": list(self.fusion_columns),
            "target_column": self.target_column,
            "target_follows_reference": True,
            "fit_source": "training_split_only",
            "heldout_used_for_fitting": False,
            "reference_operator": type(self.reference).__name__,
            "witness": self.witness.metadata(),
        }
