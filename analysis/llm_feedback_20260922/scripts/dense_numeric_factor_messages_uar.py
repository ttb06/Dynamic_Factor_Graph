#!/usr/bin/env python3
"""Dense trajectory messages over verified numerical constraint graphs.

Binary repair uses a graph only after a row violates an executable predicate.
This mechanism also uses *satisfied* readable factors: a train-fitted
conditional potential sends a message between adjacent numerical nodes at
middle reverse steps.  Two transports are provided:

``residual``
    shrink a node toward its cross-validated conditional prediction;
``copula``
    change row assignments (ranks) according to the conditional prediction
    while preserving the current batch marginal exactly.

The latter separates two objectives that are otherwise easy to confound:
constraint/joint improvement and marginal distortion.  A vectorised typed
projection can subsequently handle actual violations.  No held-out row is
used to fit a factor or choose its reliability.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

from conditioned_tabdiff_adapter import decode, seed_all
from batch_distribution_transport import (
    BatchDistributionTransport, ConditionalBatchDistributionTransport,
    QuantileConditionalBatchDistributionTransport, numeric_columns,
)
from posterior_quantile_transport import (
    PosteriorQuantileConditionalBatchDistributionTransport,
)
from uncertainty_aware_quantile_transport import (
    UncertaintyAwarePosteriorQuantileTransport,
)
from categorical_logit_guidance import CategoricalLogitGuidance
from knowledge_graph_protocol import predicate_mask
from subgroup_projection_online import PreviewModel
from tabdiff_probe import preprocess_dataset
from typed_dynamic_graph_guidance import edge_masks, graph_metrics


def edge_columns(edge: dict) -> set[str]:
    if edge.get("columns"):
        return set(edge["columns"])
    atoms = edge.get("applicability", []) + edge.get("requirements", [])
    columns = {atom["left_column"] for atom in atoms}
    columns |= {
        atom["right_column"] for atom in atoms
        if atom.get("right_kind") == "column" and atom.get("right_column")
    }
    return columns


class DenseNumericFactors:
    """Cross-validated conditional potentials induced by graph adjacency."""

    def __init__(self, train_data, train: pd.DataFrame, info: dict, edges: list[dict],
                 seed: int, trees: int, min_r2: float,
                 factor_usage_roles: set[str] | None = None,
                 factor_relation_families: set[str] | None = None):
        self.train_data = train_data
        self.train = train
        self.edges = edges
        self.numeric_names = [info["column_names"][i] for i in info["num_col_idx"]]
        target_names = [info["column_names"][i] for i in info["target_col_idx"]]
        self.regression_target_is_numerical = (
            info["task_type"] == "regression"
            and train_data.d_numerical == len(self.numeric_names) + len(target_names)
        )
        # TabDiff prepends regression targets to the numerical model block.
        # Graph programs still operate only on feature nodes, but correction
        # masks and re-encoding must use model coordinates (including that
        # untouched target prefix) to avoid shifting every action by one slot.
        self.model_numeric_names = (
            target_names + self.numeric_names
            if self.regression_target_is_numerical else self.numeric_names
        )
        self.numeric_set = set(self.numeric_names)
        self.num_pos = {name: index for index, name in enumerate(self.model_numeric_names)}
        self.scales = {
            name: max(float(pd.to_numeric(train[name], errors="coerce").std()), 1e-6)
            for name in self.numeric_names
        }

        neighbours: dict[str, set[str]] = defaultdict(set)
        incident: dict[str, list[dict]] = defaultdict(list)
        for edge in edges:
            if factor_usage_roles and edge.get("usage_role") not in factor_usage_roles:
                continue
            if (factor_relation_families
                    and edge.get("relation_family") not in factor_relation_families):
                continue
            columns = sorted(edge_columns(edge) & self.numeric_set)
            for target in columns:
                neighbours[target].update(column for column in columns if column != target)
                incident[target].append(edge)

        self.models: dict[str, dict] = {}
        if trees <= 0:
            return

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train))
        split = max(1, int(0.8 * len(order)))
        fit_indices, validation_indices = order[:split], order[split:]
        for ordinal, target in enumerate(sorted(neighbours)):
            predictors = sorted(neighbours[target])
            if not predictors or len(validation_indices) < 20:
                continue
            x = train[predictors].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
            y = pd.to_numeric(train[target], errors="coerce").fillna(0).to_numpy(float)
            validation_model = ExtraTreesRegressor(
                n_estimators=trees, min_samples_leaf=8, max_features=1.0,
                n_jobs=-1, random_state=seed + ordinal,
            )
            validation_model.fit(x[fit_indices], y[fit_indices])
            prediction = validation_model.predict(x[validation_indices])
            score = float(r2_score(y[validation_indices], prediction))
            residual = y[validation_indices] - prediction
            residual_scale = max(float(np.quantile(np.abs(residual), 0.75)), 0.05 * self.scales[target])
            validation_tree_predictions = np.stack(
                [tree.predict(x[validation_indices])
                 for tree in validation_model.estimators_[:min(32, trees)]], axis=0)
            uncertainty_reference = max(
                float(np.quantile(validation_tree_predictions.std(axis=0) / residual_scale, 0.90)),
                1e-6,
            )
            if not np.isfinite(score) or score < min_r2:
                continue
            model = ExtraTreesRegressor(
                n_estimators=trees, min_samples_leaf=8, max_features=1.0,
                n_jobs=-1, random_state=seed + 10_000 + ordinal,
            )
            model.fit(x, y)
            self.models[target] = {
                "model": model,
                "predictors": predictors,
                "validation_r2": score,
                "reliability": float(np.clip(score, 0.0, 1.0)),
                "residual_scale": residual_scale,
                "uncertainty_reference": uncertainty_reference,
                "incident_edges": incident[target],
            }

    def encode_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame[self.model_numeric_names].apply(
            pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        return self.train_data.num_transform.transform(matrix)

    @staticmethod
    def schedule_weight(t: float, minimum: float, maximum: float, schedule: str) -> float:
        if t < minimum or t > maximum:
            return 0.0
        progress = (maximum - t) / max(maximum - minimum, 1e-8)
        if schedule == "flat":
            return 1.0
        if schedule == "late_ramp":
            return 0.2 + 0.8 * progress
        if schedule == "middle_peak":
            return 0.2 + 0.8 * (4.0 * progress * (1.0 - progress))
        raise ValueError(schedule)

    def active_mask(self, frame: pd.DataFrame, model_spec: dict) -> np.ndarray:
        # A node may participate in several conditional factors.  The message
        # is active if at least one incident factor applies to the current row.
        result = np.zeros(len(frame), dtype=bool)
        for edge in model_spec["incident_edges"]:
            active, _, _ = edge_masks(frame, edge)
            result |= active
        return result

    def factor_status_masks(self, frame: pd.DataFrame,
                            model_spec: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return applicable and currently violating rows for a factor.

        Factor messages can also be sent on satisfied rows.  The second mask
        is used only for mechanism accounting; it does not suppress positive
        messages.  A row is marked violating if any incident typed edge is
        violated at the current clean preview.
        """
        applicable = np.zeros(len(frame), dtype=bool)
        violating = np.zeros(len(frame), dtype=bool)
        for edge in model_spec["incident_edges"]:
            edge_active, _, edge_bad = edge_masks(frame, edge)
            program = edge.get("program")
            if program:
                edge_active = self._program_active(frame, program)
                kind = program.get("kind")
                tolerance = float(program.get("tolerance", 0.0) or 0.0)
                if kind == "less_equal":
                    left = pd.to_numeric(
                        frame[program["left"]], errors="coerce").fillna(0).to_numpy(float)
                    right = pd.to_numeric(
                        frame[program["right"]], errors="coerce").fillna(0).to_numpy(float)
                    edge_bad = edge_active & (left > right + tolerance)
                elif kind == "ordered_triplet":
                    values = frame[program["columns"]].apply(
                        pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
                    edge_bad = edge_active & (
                        (values[:, 0] > values[:, 1] + tolerance)
                        | (values[:, 1] > values[:, 2] + tolerance))
                elif kind == "sum_equal":
                    values = frame[program["columns"]].apply(
                        pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
                    edge_bad = edge_active & (
                        np.abs(values.sum(axis=1) - float(program["target"])) > tolerance)
                elif kind == "abs_equal":
                    source = pd.to_numeric(
                        frame[program["source"]], errors="coerce").fillna(0).to_numpy(float)
                    target = pd.to_numeric(
                        frame[program["target_column"]], errors="coerce").fillna(0).to_numpy(float)
                    desired = np.abs(source - float(program.get("center", 0.0)))
                    edge_bad = edge_active & (np.abs(target - desired) > tolerance)
                elif kind == "zero_cooccurrence":
                    values = frame[program["columns"]].apply(
                        pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
                    zero = np.abs(values) <= tolerance
                    edge_bad = edge_active & ~(zero.all(axis=1) | (~zero).all(axis=1))
                else:
                    # Unknown soft programs remain eligible for messages but
                    # are not mislabeled as violations.
                    edge_bad = np.zeros(len(frame), dtype=bool)
            applicable |= edge_active
            violating |= edge_bad
        return applicable, applicable & violating

    @staticmethod
    def rank_transport(current: np.ndarray, score: np.ndarray, strength: float) -> np.ndarray:
        """Copula update with exact empirical-marginal preservation."""
        n = len(current)
        if n < 2:
            return current.copy()
        current_order = np.argsort(current, kind="mergesort")
        score_order = np.argsort(score, kind="mergesort")
        current_rank = np.empty(n, dtype=float); current_rank[current_order] = np.arange(n)
        score_rank = np.empty(n, dtype=float); score_rank[score_order] = np.arange(n)
        blended_rank = (1.0 - strength) * current_rank + strength * score_rank
        assignment_order = np.argsort(blended_rank, kind="mergesort")
        result = np.empty(n, dtype=float)
        result[assignment_order] = np.sort(current)
        return result

    def message_pass(self, frame: pd.DataFrame, mode: str, strength: float,
                     passes: int, clip: float, residual_gate: float = 0.0,
                     uncertainty_gate: float = -1.0,
                     uncertainty_trees: int = 16,
                     satisfied_strength_scale: float = 1.0,
                     violating_strength_scale: float = 1.0,
                     ) -> tuple[pd.DataFrame, np.ndarray, dict]:
        """Apply conditional-factor messages to selected row states.

        ``residual_gate`` is measured in cross-validated residual scales and
        activates a row only when its current clean preview disagrees enough
        with the graph-conditioned prediction. ``uncertainty_gate`` is the
        standard deviation across ExtraTrees predictions in the same units;
        a negative value disables that gate. Together these turn a globally
        applicable semantic edge into a row- and state-dependent active edge
        without asking the LLM to judge noisy intermediate rows.
        """
        result = frame.copy()
        # Decoded previews may use float32 columns.  Messages are computed in
        # float64; promote once so pandas does not silently narrow assignments.
        result[self.numeric_names] = result[self.numeric_names].apply(
            pd.to_numeric, errors="coerce").astype(np.float64)
        changed = np.zeros((len(frame), len(self.model_numeric_names)), dtype=bool)
        events = 0
        movement = 0.0
        candidate_row_factors = 0
        active_row_factors = 0
        low_residual_rejections = 0
        uncertainty_rejections = 0
        satisfied_candidate_row_factors = 0
        violating_candidate_row_factors = 0
        satisfied_active_row_factors = 0
        violating_active_row_factors = 0
        satisfied_events = 0
        violating_events = 0
        for _ in range(passes):
            for target, specification in self.models.items():
                active, violating = self.factor_status_masks(result, specification)
                indices = np.flatnonzero(active)
                if len(indices) < 2:
                    continue
                candidate_row_factors += int(len(indices))
                violating_candidate_row_factors += int(violating[indices].sum())
                satisfied_candidate_row_factors += int((~violating[indices]).sum())
                predictors = specification["predictors"]
                x = result.iloc[indices][predictors].apply(
                    pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
                predicted = specification["model"].predict(x)
                current = pd.to_numeric(result.iloc[indices][target], errors="coerce").fillna(0).to_numpy(float)
                residual_scale = max(float(specification["residual_scale"]), 1e-8)
                local_gate = np.ones(len(indices), dtype=bool)
                if residual_gate > 0:
                    residual_ok = np.abs(predicted - current) / residual_scale >= residual_gate
                    low_residual_rejections += int((local_gate & ~residual_ok).sum())
                    local_gate &= residual_ok
                if uncertainty_gate >= 0:
                    # ExtraTrees exposes a cheap epistemic-disagreement proxy:
                    # row states far from supported train regions make its
                    # individual trees disagree. Compute it only when used.
                    ensemble = specification["model"].estimators_[:max(1, uncertainty_trees)]
                    tree_predictions = np.stack(
                        [tree.predict(x) for tree in ensemble], axis=0)
                    normalized_uncertainty = (
                        tree_predictions.std(axis=0) / residual_scale
                        / float(specification["uncertainty_reference"]))
                    uncertainty_ok = normalized_uncertainty <= uncertainty_gate
                    uncertainty_rejections += int((local_gate & ~uncertainty_ok).sum())
                    local_gate &= uncertainty_ok
                indices = indices[local_gate]
                predicted = predicted[local_gate]
                current = current[local_gate]
                local_violating = violating[indices]
                active_row_factors += int(len(indices))
                violating_active_row_factors += int(local_violating.sum())
                satisfied_active_row_factors += int((~local_violating).sum())
                if len(indices) < 2:
                    continue
                effective = float(np.clip(strength * specification["reliability"], 0.0, 1.0))
                if effective <= 0:
                    continue
                if mode == "copula":
                    # Satisfied states carry positive dependence information,
                    # but they vastly outnumber violations in realistic
                    # trajectories.  Transport the two status groups
                    # separately so their weights can be calibrated without
                    # letting the majority group erase corrective messages.
                    updated = current.copy()
                    for status, scale in (
                        (~local_violating, satisfied_strength_scale),
                        (local_violating, violating_strength_scale),
                    ):
                        if int(status.sum()) >= 2 and scale > 0:
                            updated[status] = self.rank_transport(
                                current[status], predicted[status],
                                float(np.clip(effective * scale, 0.0, 1.0)))
                elif mode == "residual":
                    delta = predicted - current
                    # A factor should improve dependence rather than shift the
                    # marginal location.  Remove the batch mean and cap by a
                    # cross-validated residual scale.
                    delta -= float(delta.mean())
                    bound = clip * float(specification["residual_scale"])
                    status_scale = np.where(
                        local_violating, violating_strength_scale,
                        satisfied_strength_scale)
                    updated = current + effective * status_scale * np.clip(
                        delta, -bound, bound)
                    # Preserve first two moments of the current clean preview.
                    old_std = float(current.std()); new_std = float(updated.std())
                    if old_std > 1e-8 and new_std > 1e-8:
                        updated = (updated - updated.mean()) * (old_std / new_std) + current.mean()
                else:
                    raise ValueError(mode)
                delta = updated - current
                local_changed = np.abs(delta) > 8 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(current))
                if local_changed.any():
                    selected = indices[local_changed]
                    result.loc[result.index[selected], target] = updated[local_changed]
                    changed[selected, self.num_pos[target]] = True
                    events += int(local_changed.sum())
                    violating_events += int(local_violating[local_changed].sum())
                    satisfied_events += int((~local_violating[local_changed]).sum())
                    movement += float(np.abs(delta[local_changed] / self.scales[target]).sum())
        return result, changed, {
            "factor_cell_events": events,
            "standardized_movement": movement,
            "factor_candidate_row_factors": candidate_row_factors,
            "factor_active_row_factors": active_row_factors,
            "factor_low_residual_rejections": low_residual_rejections,
            "factor_uncertainty_rejections": uncertainty_rejections,
            "factor_satisfied_candidate_row_factors": satisfied_candidate_row_factors,
            "factor_violating_candidate_row_factors": violating_candidate_row_factors,
            "factor_satisfied_active_row_factors": satisfied_active_row_factors,
            "factor_violating_active_row_factors": violating_active_row_factors,
            "factor_satisfied_cell_events": satisfied_events,
            "factor_violating_cell_events": violating_events,
            "factor_active_fraction": (
                active_row_factors / candidate_row_factors if candidate_row_factors else 0.0),
        }

    def row_energy(self, frame: pd.DataFrame) -> np.ndarray:
        """Train-fitted conditional energy used only as an action veto.

        Energy is normalized by the retained factors' cross-validated
        reliability.  It never uses the generation evaluator or held-out rows.
        """
        energy = np.zeros(len(frame), dtype=float)
        normalizer = 0.0
        for target, specification in self.models.items():
            predictors = specification["predictors"]
            x = frame[predictors].apply(
                pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
            observed = pd.to_numeric(
                frame[target], errors="coerce").fillna(0).to_numpy(float)
            predicted = specification["model"].predict(x)
            scale = max(float(specification["residual_scale"]), 1e-8)
            weight = float(specification["reliability"])
            energy += weight * np.minimum(((observed - predicted) / scale) ** 2, 25.0)
            normalizer += weight
        return energy / max(normalizer, 1e-8)

    def project_violations(self, frame: pd.DataFrame, strength: float,
                           margin: float = 1e-4,
                           edge_weights: dict[str, float] | None = None,
                           edge_row_masks: dict[str, np.ndarray] | None = None,
                           edge_signal_overrides: dict[str, np.ndarray] | None = None,
                           ) -> tuple[pd.DataFrame, np.ndarray, int]:
        """Vectorised standardized balanced projection for column inequalities."""
        result = frame.copy()
        result[self.numeric_names] = result[self.numeric_names].apply(
            pd.to_numeric, errors="coerce").astype(np.float64)
        changed = np.zeros((len(frame), len(self.model_numeric_names)), dtype=bool)
        events = 0
        for edge in self.edges:
            local_strength = strength * (
                float(edge_weights.get(edge["candidate_id"], 0.0))
                if edge_weights is not None else 1.0
            )
            local_strength *= float(edge.get("guidance_weight", 1.0))
            if local_strength <= 0:
                continue
            active_override = None
            if edge_row_masks is not None:
                active_override = edge_row_masks.get(edge["candidate_id"])
            signal_override = None
            if edge_signal_overrides is not None:
                signal_override = edge_signal_overrides.get(edge["candidate_id"])
            if edge.get("program"):
                program_events, program_changed = self._project_program(
                    result, edge["program"], local_strength, margin,
                    active_override=active_override,
                    signal_override=signal_override)
                changed |= program_changed
                events += program_events
                continue
            active, _, _ = edge_masks(result, edge)
            if active_override is not None:
                active &= np.asarray(active_override, dtype=bool)
            for atom in edge.get("requirements", []):
                left, right = atom["left_column"], atom.get("right_column")
                operator = atom.get("operator")
                if left not in self.numeric_set:
                    continue
                left_value = pd.to_numeric(result[left], errors="coerce").to_numpy(float)
                if atom.get("right_kind") == "column":
                    if right not in self.numeric_set:
                        continue
                    right_value = pd.to_numeric(
                        result[right], errors="coerce").to_numpy(float)
                    right_scale = self.scales[right]
                elif atom.get("right_kind") == "literal":
                    try:
                        literal = float(atom.get("right_value"))
                    except (TypeError, ValueError):
                        continue
                    right_value = np.full(len(result), literal, dtype=float)
                    right_scale = 0.0
                else:
                    continue
                if operator in {"<=", "<"}:
                    gap = left_value - right_value + (margin if operator == "<" else 0.0)
                    bad = active & (gap > 0)
                    direction = -1.0
                elif operator in {">=", ">"}:
                    gap = right_value - left_value + (margin if operator == ">" else 0.0)
                    bad = active & (gap > 0)
                    direction = 1.0
                elif operator == "==":
                    gap = np.abs(left_value - right_value) - float(atom.get("tolerance", 0.0) or 0.0)
                    bad = active & (gap > 0)
                    direction = np.sign(right_value - left_value)
                else:
                    continue
                if not bad.any():
                    continue
                sl2, sr2 = self.scales[left] ** 2, right_scale ** 2
                correction = local_strength * np.maximum(gap, 0.0)
                if np.isscalar(direction):
                    left_delta = direction * correction * sl2 / (sl2 + sr2)
                    right_delta = -direction * correction * sr2 / (sl2 + sr2)
                else:
                    left_delta = direction * correction * sl2 / (sl2 + sr2)
                    right_delta = -direction * correction * sr2 / (sl2 + sr2)
                result.loc[bad, left] = left_value[bad] + left_delta[bad]
                changed[bad, self.num_pos[left]] = True
                if atom.get("right_kind") == "column":
                    result.loc[bad, right] = right_value[bad] + right_delta[bad]
                    changed[bad, self.num_pos[right]] = True
                events += int(bad.sum())
        return result, changed, events

    def _program_active(self, frame: pd.DataFrame, program: dict) -> np.ndarray:
        active = np.ones(len(frame), dtype=bool)
        gate_columns = program.get("gate_sum_columns", [])
        if gate_columns:
            values = frame[gate_columns].apply(
                pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
            active &= values.sum(axis=1) > float(program.get("gate_sum_gt", 0.0))
        return active

    def _project_program(self, frame: pd.DataFrame, program: dict,
                         strength: float, margin: float,
                         active_override: np.ndarray | None = None,
                         signal_override: np.ndarray | None = None,
                         ) -> tuple[int, np.ndarray]:
        """Project a small, typed family of documented functional hyperedges.

        This is intentionally not a general expression evaluator: free-form
        LLM text can never reach the actuator without compilation.
        """
        changed = np.zeros((len(frame), len(self.model_numeric_names)), dtype=bool)
        active = self._program_active(frame, program)
        if active_override is not None:
            if len(active_override) != len(frame):
                raise ValueError("program active mask has the wrong number of rows")
            active &= np.asarray(active_override, dtype=bool)
        kind = program["kind"]
        transport = program.get("transport", "direct")
        tolerance = float(program.get("tolerance", margin))

        if kind == "integer":
            # LCATS-style integer-domain primitive, implemented as a typed
            # clean-preview projection rather than a free-form expression.
            # A fractional strength is useful in middle steps; the existing
            # certifier applies strength=1 at the endpoint when requested.
            column = program["column"]
            values = pd.to_numeric(
                frame[column], errors="coerce").fillna(0).to_numpy(float)
            desired = np.rint(values)
            residual = desired - values
            rows = active & (np.abs(residual) > tolerance)
            if rows.any():
                updated = values + strength * residual
                frame.loc[rows, column] = updated[rows]
                changed[rows, self.num_pos[column]] = True
            return int(rows.sum()), changed

        if kind == "sum_equal":
            columns = program["columns"]
            values = frame[columns].apply(
                pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
            if transport == "rank":
                # Preserve the target column's empirical marginal exactly and
                # use the functional program only to improve its copula with
                # the remaining endpoints.
                target_column = columns[-1]
                current = values[:, -1]
                desired = (np.asarray(signal_override, dtype=float)
                           if signal_override is not None else
                           float(program["target"]) - values[:, :-1].sum(axis=1))
                indices = np.flatnonzero(active)
                updated = current.copy()
                updated[indices] = self.rank_transport(
                    current[indices], desired[indices], strength)
                rows = active & (np.abs(updated - current) > 8 * np.finfo(np.float32).eps)
                if rows.any():
                    frame.loc[rows, target_column] = updated[rows]
                    changed[rows, self.num_pos[target_column]] = True
                return int(rows.sum()), changed
            residual = float(program["target"]) - values.sum(axis=1)
            rows = active & (np.abs(residual) > tolerance)
            if rows.any():
                weights = np.asarray(
                    [self.scales[column] ** 2 for column in columns], dtype=float)
                weights /= weights.sum()
                updated = values + strength * residual[:, None] * weights[None, :]
                for position, column in enumerate(columns):
                    frame.loc[rows, column] = updated[rows, position]
                    changed[rows, self.num_pos[column]] = True
            return int(rows.sum()), changed

        if kind == "abs_equal":
            source, target = program["source"], program["target_column"]
            source_values = pd.to_numeric(
                frame[source], errors="coerce").fillna(0).to_numpy(float)
            target_values = pd.to_numeric(
                frame[target], errors="coerce").fillna(0).to_numpy(float)
            desired = (np.asarray(signal_override, dtype=float)
                       if signal_override is not None else
                       np.abs(source_values - float(program.get("center", 0.0))))
            if transport == "rank":
                indices = np.flatnonzero(active)
                updated = target_values.copy()
                updated[indices] = self.rank_transport(
                    target_values[indices], desired[indices], strength)
                rows = active & (
                    np.abs(updated - target_values)
                    > 8 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(target_values)))
                if rows.any():
                    frame.loc[rows, target] = updated[rows]
                    changed[rows, self.num_pos[target]] = True
                return int(rows.sum()), changed
            rows = active & (np.abs(target_values - desired) > tolerance)
            if rows.any():
                updated = target_values + strength * (desired - target_values)
                frame.loc[rows, target] = updated[rows]
                changed[rows, self.num_pos[target]] = True
            return int(rows.sum()), changed

        if kind == "less_equal":
            # Minimum standardized displacement to the half-space
            # left <= right.  This is the tabular analogue of a constraint
            # projection layer; unlike a factor message it only touches rows
            # that actually violate the verified inequality.
            left, right = program["left"], program["right"]
            left_values = pd.to_numeric(
                frame[left], errors="coerce").fillna(0).to_numpy(float)
            right_values = pd.to_numeric(
                frame[right], errors="coerce").fillna(0).to_numpy(float)
            gap = left_values - right_values - tolerance
            rows = active & (gap > 0)
            if rows.any():
                sl2, sr2 = self.scales[left] ** 2, self.scales[right] ** 2
                correction = strength * np.maximum(gap, 0.0)
                frame.loc[rows, left] = (
                    left_values[rows] - correction[rows] * sl2 / (sl2 + sr2))
                frame.loc[rows, right] = (
                    right_values[rows] + correction[rows] * sr2 / (sl2 + sr2))
                changed[rows, self.num_pos[left]] = True
                changed[rows, self.num_pos[right]] = True
            return int(rows.sum()), changed

        if kind == "ordered_triplet":
            columns = program["columns"]
            if transport == "rank":
                left = pd.to_numeric(frame[columns[0]], errors="coerce").fillna(0).to_numpy(float)
                middle = pd.to_numeric(frame[columns[1]], errors="coerce").fillna(0).to_numpy(float)
                right = pd.to_numeric(frame[columns[2]], errors="coerce").fillna(0).to_numpy(float)
                desired = (np.asarray(signal_override, dtype=float)
                           if signal_override is not None else (left + right) / 2.0)
                indices = np.flatnonzero(active)
                updated = middle.copy()
                updated[indices] = self.rank_transport(middle[indices], desired[indices], strength)
                rows = active & (
                    np.abs(updated - middle)
                    > 8 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(middle)))
                if rows.any():
                    frame.loc[rows, columns[1]] = updated[rows]
                    changed[rows, self.num_pos[columns[1]]] = True
                return int(rows.sum()), changed
            event_rows = np.zeros(len(frame), dtype=bool)
            # Two PAVA-like sweeps suffice for a three-node chain.  Each pair
            # uses the same standardized minimum-displacement geometry as the
            # public inequality layer.
            for _ in range(2):
                for left, right in zip(columns[:-1], columns[1:]):
                    left_values = pd.to_numeric(
                        frame[left], errors="coerce").fillna(0).to_numpy(float)
                    right_values = pd.to_numeric(
                        frame[right], errors="coerce").fillna(0).to_numpy(float)
                    gap = left_values - right_values
                    rows = active & (gap > tolerance)
                    if not rows.any():
                        continue
                    sl2, sr2 = self.scales[left] ** 2, self.scales[right] ** 2
                    correction = strength * np.maximum(gap, 0.0)
                    frame.loc[rows, left] = (
                        left_values[rows] - correction[rows] * sl2 / (sl2 + sr2))
                    frame.loc[rows, right] = (
                        right_values[rows] + correction[rows] * sr2 / (sl2 + sr2))
                    changed[rows, self.num_pos[left]] = True
                    changed[rows, self.num_pos[right]] = True
                    event_rows |= rows
            return int(event_rows.sum()), changed

        raise ValueError(f"unsupported typed graph program: {kind}")

    def program_signal(self, frame: pd.DataFrame, edge: dict
                       ) -> tuple[np.ndarray, float] | None:
        """Return the row-level clean-state signal consumed by a program.

        Consecutive reverse states can disagree strongly even after the edge's
        coarse time window opens.  This signal lets the sampler activate an
        edge only for rows whose proposed semantic target has become stable.
        The scale is fitted from train, never from held-out/evaluation data.
        """
        program = edge.get("program")
        if not program:
            requirements = edge.get("requirements", [])
            if len(requirements) != 1:
                return None
            requirement = requirements[0]
            left = requirement.get("left_column")
            if left not in self.numeric_set:
                return None
            left_values = pd.to_numeric(
                frame[left], errors="coerce").fillna(0).to_numpy(float)
            if requirement.get("right_kind") == "column":
                right = requirement.get("right_column")
                if right not in self.numeric_set:
                    return None
                right_values = pd.to_numeric(
                    frame[right], errors="coerce").fillna(0).to_numpy(float)
                return left_values - right_values, math.sqrt(
                    self.scales[left] ** 2 + self.scales[right] ** 2)
            if requirement.get("right_kind") == "literal":
                try:
                    target = float(requirement.get("right_value"))
                except (TypeError, ValueError):
                    return None
                return left_values - target, self.scales[left]
            return None
        kind = program["kind"]
        if kind == "integer":
            column = program["column"]
            values = pd.to_numeric(
                frame[column], errors="coerce").fillna(0).to_numpy(float)
            # Stability is defined by agreement on the nearest grid point,
            # not by the noisy fractional residual itself.
            return np.rint(values), 1.0
        if kind == "ordered_triplet":
            columns = program["columns"]
            left = pd.to_numeric(frame[columns[0]], errors="coerce").fillna(0).to_numpy(float)
            right = pd.to_numeric(frame[columns[2]], errors="coerce").fillna(0).to_numpy(float)
            return (left + right) / 2.0, self.scales[columns[1]]
        if kind == "sum_equal":
            columns = program["columns"]
            values = frame[columns[:-1]].apply(
                pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
            return float(program["target"]) - values.sum(axis=1), self.scales[columns[-1]]
        if kind == "abs_equal":
            source, target = program["source"], program["target_column"]
            values = pd.to_numeric(frame[source], errors="coerce").fillna(0).to_numpy(float)
            return np.abs(values - float(program.get("center", 0.0))), self.scales[target]
        if kind == "less_equal":
            left, right = program["left"], program["right"]
            left_values = pd.to_numeric(frame[left], errors="coerce").fillna(0).to_numpy(float)
            right_values = pd.to_numeric(frame[right], errors="coerce").fillna(0).to_numpy(float)
            return left_values - right_values, math.sqrt(
                self.scales[left] ** 2 + self.scales[right] ** 2)
        return None

    @staticmethod
    def stable_signal_mask(current: np.ndarray, previous: np.ndarray | None,
                           threshold: float, mode: str, scale: float) -> np.ndarray:
        if previous is None:
            return np.zeros(len(current), dtype=bool)
        if mode == "rank":
            n = len(current)
            current_order = np.argsort(current, kind="mergesort")
            previous_order = np.argsort(previous, kind="mergesort")
            current_rank = np.empty(n, dtype=float)
            previous_rank = np.empty(n, dtype=float)
            current_rank[current_order] = np.arange(n)
            previous_rank[previous_order] = np.arange(n)
            return np.abs(current_rank - previous_rank) / max(n - 1, 1) <= threshold
        return np.abs(current - previous) <= threshold * max(float(scale), 1e-8)

    def certify(self, frame: pd.DataFrame, maximum_passes: int = 4) -> tuple[pd.DataFrame, dict]:
        """Deterministic final projection for this numerical graph.

        Repeating the balanced projection handles connected inequalities where
        repairing one endpoint can activate a neighbouring edge.  Unlike the
        older categorical post-repair pilot, this supports the numerical
        column-to-column relations used by the News public graph.
        """
        result = frame.copy(); total_events = 0; passes = 0
        changed_rows = np.zeros(len(result), dtype=bool)
        for _ in range(maximum_passes):
            result, changed, events = self.project_violations(result, 1.0)
            passes += 1; total_events += events
            changed_rows |= changed.any(axis=1)
            if events == 0:
                break
        return result, {
            "action_events": int(total_events),
            "unique_rows_actuated": int(changed_rows.sum()),
            "projection_passes": passes,
        }

    def metadata(self) -> list[dict]:
        return [
            {"target": target, "predictors": spec["predictors"],
             "validation_r2": spec["validation_r2"],
             "reliability": spec["reliability"],
             "residual_scale": spec["residual_scale"],
             "uncertainty_reference": spec["uncertainty_reference"],
             "incident_edges": [edge["candidate_id"] for edge in spec["incident_edges"]]}
            for target, spec in self.models.items()
        ]


class BenefitGate:
    """Train-only row/edge action gate learned from forward-noised clean rows."""

    def __init__(self, manifest_path: Path, allowed_edge_ids: set[str] | None = None):
        manifest = json.loads(manifest_path.read_text())
        self.feature_names = manifest["feature_names"]
        self.proposal_strength = float(manifest.get("proposal_strength", .50))
        self.allowed_edge_ids = allowed_edge_ids
        self.entries = {}
        for edge_id, specification in manifest["models"].items():
            self.entries[edge_id] = {
                "model": joblib.load(manifest_path.parent / specification["path"]),
                "threshold": float(specification["threshold"]),
            }

    def row_masks(self, frame: pd.DataFrame, preview_numeric: torch.Tensor,
                  sigma_numeric: torch.Tensor, t: float,
                  factors: DenseNumericFactors) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        masks: dict[str, np.ndarray] = {}
        proposed_rows = 0
        accepted_rows = 0
        original_edges = factors.edges
        try:
            for edge in original_edges:
                edge_id = edge["candidate_id"]
                if self.allowed_edge_ids is not None and edge_id not in self.allowed_edge_ids:
                    continue
                entry = self.entries.get(edge_id)
                if entry is None:
                    continue
                proposal_edge = copy.deepcopy(edge)
                proposal_edge["guidance_weight"] = 1.0
                factors.edges = [proposal_edge]
                proposed, changed, _ = factors.project_violations(
                    frame, self.proposal_strength)
                changed_tensor = torch.as_tensor(
                    changed, dtype=torch.bool, device=preview_numeric.device)
                changed_rows = changed_tensor.any(dim=1)
                proposed_rows += int(changed_rows.sum())
                accepted = np.zeros(len(frame), dtype=bool)
                if changed_rows.any():
                    target = torch.as_tensor(
                        factors.encode_numeric(proposed), dtype=preview_numeric.dtype,
                        device=preview_numeric.device)
                    mask_float = changed_tensor.to(preview_numeric.dtype)
                    changed_count = mask_float.sum(dim=1).clamp_min(1.0)
                    delta = target - preview_numeric
                    sigma = sigma_numeric
                    if sigma.ndim == 1:
                        sigma = sigma.unsqueeze(0).expand(len(frame), -1)
                    features = torch.stack([
                        torch.full_like(changed_count, t),
                        (delta.abs() * mask_float).sum(dim=1) / changed_count,
                        ((delta.square() * mask_float).sum(dim=1) / changed_count).sqrt(),
                        (preview_numeric.abs() * mask_float).sum(dim=1) / changed_count,
                        (sigma * mask_float).sum(dim=1) / changed_count,
                        preview_numeric.abs().mean(dim=1),
                        preview_numeric.std(dim=1),
                    ], dim=1)
                    selected_features = features[changed_rows].detach().cpu().numpy()
                    probability = entry["model"].predict_proba(selected_features)[:, 1]
                    selected = probability >= entry["threshold"]
                    indices = np.flatnonzero(changed_rows.detach().cpu().numpy())
                    accepted[indices[selected]] = True
                masks[edge_id] = accepted
                accepted_rows += int(accepted.sum())
        finally:
            factors.edges = original_edges
        return masks, {
            "benefit_gate_rows_proposed": proposed_rows,
            "benefit_gate_rows_accepted": accepted_rows,
        }


@torch.no_grad()
def sample(diffusion, preview, train_data, info: dict, factors: DenseNumericFactors,
           samples: int, mode: str, strength: float, passes: int, clip: float,
           factor_residual_gate: float, factor_uncertainty_gate: float,
           factor_uncertainty_trees: int,
           factor_satisfied_strength_scale: float,
           factor_violating_strength_scale: float,
           factor_min_t: float, factor_max_t: float, factor_schedule: str,
           interval_space: str, projection_strength: float,
           projection_min_t: float, projection_max_t: float,
           projection_interval_space: str,
           projection_schedule: str,
           projection_trust_radius: float,
           projection_density_veto: bool,
           projection_density_slack: float,
           projection_stability_threshold: float,
           projection_stability_mode: str,
           projection_route_t: float,
           projection_route_mode: str,
           benefit_gate: BenefitGate | None = None,
           stochastic_sampler: bool = False,
           second_order_correction: bool = False,
           guide_second_order: bool = False,
           guidance_stage: str = "predictor",
           projection_persistence_steps: int = 1,
           batch_transport: (BatchDistributionTransport
                             | ConditionalBatchDistributionTransport
                             | QuantileConditionalBatchDistributionTransport
                             | PosteriorQuantileConditionalBatchDistributionTransport
                             | UncertaintyAwarePosteriorQuantileTransport
                             | None) = None,
           transport_strength: float = 0.0,
           transport_min_t: float = 0.0,
           transport_max_t: float = 0.0,
           transport_schedule: str = "flat",
           transport_once: bool = False,
           transport_max_calls: int = 0,
           transport_min_call_gap: float = 0.0,
           categorical_guidance: CategoricalLogitGuidance | None = None,
           categorical_guidance_strength: float = 0.0,
           categorical_guidance_min_t: float = 0.0,
           categorical_guidance_max_t: float = 0.0,
           categorical_guidance_schedule: str = "flat") -> tuple[torch.Tensor, dict]:
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num = diffusion.num_schedule.total_noise(times)
    sigma_cat = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num); sigma_num_next[1:] = sigma_num[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat); sigma_cat_next[1:] = sigma_cat[:-1]
    if stochastic_sampler:
        # Match TabDiff's released EDM sampler.  Keeping this construction in
        # the guided path makes a zero-action run exactly comparable to the
        # official stochastic baseline rather than to a first-order proxy.
        gamma = min(1.0 / diffusion.num_timesteps, math.sqrt(2.0) - 1.0) * torch.ones_like(sigma_num)
        sigma_num_hat = sigma_num + gamma * sigma_num
        t_hat = diffusion.num_schedule.inverse_to_t(sigma_num_hat)
        t_hat = torch.min(t_hat, dim=-1, keepdim=True).values
        out_of_bound = (t_hat > 1).squeeze(-1)
        sigma_num_hat[out_of_bound] = sigma_num[out_of_bound]
        t_hat[out_of_bound] = times[out_of_bound]
        sigma_cat_hat = diffusion.cat_schedule.total_noise(t_hat)
    else:
        sigma_num_hat = sigma_num
        sigma_cat_hat = sigma_cat
        t_hat = times
    # Image-space sigma values are not numerically transferable to normalized
    # tabular columns.  A log-sigma quantile is comparable across learned,
    # feature-specific schedules: zero is the clean end and one the noisy end.
    representative_sigma = sigma_num.float().median(dim=1).values.clamp_min(1e-12)
    log_sigma = representative_sigma.log()
    noise_quantile = (log_sigma - log_sigma.min()) / (log_sigma.max() - log_sigma.min()).clamp_min(1e-12)
    # For absorbing categorical diffusion, the expected masked fraction is a
    # more faithful readability coordinate than log(sigma).  The exact clean
    # endpoint has sigma=0; including log(0) in a min/max normalization used
    # to compress every non-final state above ~0.55 and silently left ordinary
    # middle intervals empty.  Mask probability is bounded, monotone and has
    # the direct interpretation "fraction of cells still erased".
    categorical_mask_probability = (-torch.expm1(-sigma_cat.float())).median(
        dim=1).values
    categorical_noise_quantile = categorical_mask_probability / (
        categorical_mask_probability.max().clamp_min(1e-12))
    feature_log_sigma = sigma_num.float().clamp_min(1e-12).log()
    feature_noise_quantile = (
        (feature_log_sigma - feature_log_sigma.min(dim=0).values)
        / (feature_log_sigma.max(dim=0).values - feature_log_sigma.min(dim=0).values).clamp_min(1e-12)
    )
    numeric = torch.randn((samples, diffusion.num_numerical_features), device=device) * sigma_num[-1]
    categorical = diffusion._sample_masked_prior(samples, len(diffusion.num_classes))
    trace = []
    totals = defaultdict(float)
    previous_program_signals: dict[str, np.ndarray] = {}
    stable_streaks: dict[str, np.ndarray] = {}
    routed_program_signals: dict[str, np.ndarray] = {}
    routed_program_masks: dict[str, np.ndarray] = {}
    route_captured = False
    # In auto_knee mode we observe the readability of the graph before the
    # late actuation window.  Each snapshot stores row-level stable masks and
    # typed program signals.  At the first actuation step, the snapshot at the
    # steepest increase of the monotone stability curve is selected.  This
    # replaces a dataset-tuned route timestep with a trajectory statistic.
    route_candidates: list[dict] = []
    transport_applied = False
    transport_calls = 0
    last_transport_coordinate: float | None = None

    for index in reversed(range(diffusion.num_timesteps)):
        t = float(times[index])
        numeric_hat = numeric + (
            sigma_num_hat[index].square() - sigma_num[index].square()
        ).clamp_min(0).sqrt() * torch.randn_like(numeric)
        move_chance = -torch.expm1(sigma_cat[index] - sigma_cat_hat[index])
        categorical_hat, _ = diffusion.q_xt(categorical, move_chance)
        categorical_oh = diffusion.to_one_hot(categorical_hat).to(numeric.dtype)
        denoised, raw_logits = preview.base(
            numeric_hat.float(), categorical_oh, t_hat[index].squeeze().repeat(samples),
            sigma=sigma_num_hat[index].unsqueeze(0).repeat(samples, 1),
        )
        sigma_q = float(noise_quantile[index])
        categorical_sigma_q = float(categorical_noise_quantile[index])
        categorical_weight = factors.schedule_weight(
            categorical_sigma_q, categorical_guidance_min_t, categorical_guidance_max_t,
            categorical_guidance_schedule) if categorical_guidance is not None else 0.0
        categorical_diagnostics = {}
        if (categorical_guidance is not None
                and categorical_guidance_strength > 0 and categorical_weight > 0):
            raw_logits, categorical_diagnostics = categorical_guidance.apply(
                raw_logits, categorical_hat,
                categorical_guidance_strength * categorical_weight)
        codes = preview.predicted_codes(categorical_oh, raw_logits)
        frame = decode(train_data, torch.cat([denoised, codes.float()], dim=1).cpu(), info)
        corrected = frame.copy()
        mask = np.zeros((samples, len(factors.model_numeric_names)), dtype=bool)
        current_program_signals: dict[str, tuple[np.ndarray, float]] = {}
        for edge in factors.edges:
            signal = factors.program_signal(frame, edge)
            if signal is not None:
                current_program_signals[edge["candidate_id"]] = signal
        current_stable_masks: dict[str, np.ndarray] = {}
        for edge_id, (signal, scale) in current_program_signals.items():
            one_step = factors.stable_signal_mask(
                signal, previous_program_signals.get(edge_id),
                projection_stability_threshold, projection_stability_mode, scale)
            prior_streak = stable_streaks.get(
                edge_id, np.zeros(samples, dtype=np.int16))
            streak = np.where(one_step, prior_streak + 1, 0).astype(np.int16)
            stable_streaks[edge_id] = streak
            current_stable_masks[edge_id] = (
                streak >= max(int(projection_persistence_steps), 1))
        interval_coordinate = t if interval_space == "time" else sigma_q
        step = {"t": t, "representative_sigma": float(representative_sigma[index]),
                "log_sigma_quantile": sigma_q, "interval_coordinate": interval_coordinate,
                "categorical_mask_probability": float(categorical_mask_probability[index]),
                "factor_cell_events": 0, "projection_events": 0,
                "standardized_movement": 0.0}
        step["categorical_guidance_schedule_weight"] = categorical_weight
        step["categorical_log_sigma_quantile"] = categorical_sigma_q
        step.update(categorical_diagnostics)

        schedule_weight = factors.schedule_weight(
            interval_coordinate, factor_min_t, factor_max_t, factor_schedule)
        step["factor_schedule_weight"] = schedule_weight
        if schedule_weight > 0 and strength > 0 and factors.models:
            corrected, factor_mask, diagnostics = factors.message_pass(
                corrected, mode, strength * schedule_weight, passes, clip,
                residual_gate=factor_residual_gate,
                uncertainty_gate=factor_uncertainty_gate,
                uncertainty_trees=factor_uncertainty_trees,
                satisfied_strength_scale=factor_satisfied_strength_scale,
                violating_strength_scale=factor_violating_strength_scale)
            mask |= factor_mask
            step.update(diagnostics)
        transport_coordinate = sigma_q
        transport_weight = factors.schedule_weight(
            transport_coordinate, transport_min_t, transport_max_t,
            transport_schedule) if batch_transport is not None else 0.0
        step["transport_schedule_weight"] = transport_weight
        call_limit = 1 if transport_once else max(int(transport_max_calls), 0)
        call_budget_available = call_limit == 0 or transport_calls < call_limit
        call_gap_satisfied = (
            last_transport_coordinate is None
            or abs(transport_coordinate - last_transport_coordinate)
            >= max(float(transport_min_call_gap), 0.0)
        )
        if (batch_transport is not None and transport_strength > 0
                and transport_weight > 0 and call_budget_available and call_gap_satisfied):
            corrected, transport_changed, transport_diagnostics = batch_transport.apply(
                corrected, min(1.0, transport_strength * transport_weight))
            for local_index, column in enumerate(batch_transport.columns):
                if column in factors.num_pos:
                    mask[:, factors.num_pos[column]] |= transport_changed[:, local_index]
            step.update(transport_diagnostics)
            transport_applied = True
            transport_calls += 1
            last_transport_coordinate = transport_coordinate
            step["transport_call_index"] = transport_calls
        edge_weights = None
        if projection_interval_space == "edge_log_sigma_quantile":
            edge_weights = {}
            edge_coordinates = {}
            for edge in factors.edges:
                positions = [factors.num_pos[column] for column in edge_columns(edge)
                             if column in factors.num_pos]
                coordinate = float(feature_noise_quantile[index, positions].median()) if positions else sigma_q
                edge_coordinates[edge["candidate_id"]] = coordinate
                edge_weights[edge["candidate_id"]] = factors.schedule_weight(
                    coordinate, projection_min_t, projection_max_t,
                    projection_schedule)
            projection_coordinate = float(np.mean(list(edge_coordinates.values()))) if edge_coordinates else sigma_q
            projection_weight = max(edge_weights.values(), default=0.0)
            step["projection_active_edges"] = int(sum(value > 0 for value in edge_weights.values()))
            step["projection_edge_coordinates"] = edge_coordinates
        else:
            projection_coordinate = t if projection_interval_space == "time" else sigma_q
            projection_weight = factors.schedule_weight(
                projection_coordinate, projection_min_t, projection_max_t,
                projection_schedule)
        step["projection_interval_coordinate"] = projection_coordinate
        step["projection_schedule_weight"] = projection_weight
        automatic_route_modes = {
            "auto_knee", "auto_edge_knee", "auto_isotonic", "auto_edge_isotonic",
            "auto_valley_isotonic", "auto_edge_valley_isotonic",
        }
        if projection_route_mode in automatic_route_modes and not route_captured:
            candidate_masks = current_stable_masks
            counts = [int(value.sum()) for value in candidate_masks.values()]
            stable_fraction = float(sum(counts) / max(samples * len(counts), 1))
            route_candidates.append({
                "coordinate": float(projection_coordinate),
                "signals": {edge_id: signal.copy()
                            for edge_id, (signal, _) in current_program_signals.items()},
                "masks": {edge_id: value.copy()
                          for edge_id, value in candidate_masks.items()},
                "stable_fraction": stable_fraction,
            })
            step["projection_auto_candidate_fraction"] = stable_fraction
            # Guidance is not applied before projection_weight becomes
            # positive.  We can therefore use the whole noisy-to-middle prefix
            # to locate the readability transition without a second sampler
            # pass or access to final test metrics.
            if projection_weight > 0 and len(route_candidates) >= 3:
                coordinates = np.asarray(
                    [item["coordinate"] for item in route_candidates], dtype=float)
                delta_coordinate = np.abs(np.diff(coordinates)).clip(min=1e-12)
                selected_coordinates = {}
                selected_slopes = {}
                edge_specific = projection_route_mode in {
                    "auto_edge_knee", "auto_edge_isotonic",
                    "auto_edge_valley_isotonic"
                }
                robust_isotonic = projection_route_mode in {
                    "auto_isotonic", "auto_edge_isotonic",
                    "auto_valley_isotonic", "auto_edge_valley_isotonic"
                }
                valley_anchored = projection_route_mode in {
                    "auto_valley_isotonic", "auto_edge_valley_isotonic"
                }

                def fitted_readability(values: np.ndarray) -> np.ndarray:
                    if robust_isotonic:
                        # Pool transient reversals instead of letting a single
                        # high-noise spike dominate the derivative.  The fit
                        # is monotone in trajectory order (noisy -> clean) and
                        # uses no target data or final sample metric.
                        return IsotonicRegression(increasing=True).fit_transform(
                            np.arange(len(values), dtype=float), values)
                    return np.maximum.accumulate(values)

                def select_transition(values: np.ndarray) -> tuple[int, float]:
                    # The first candidate has no preceding preview and its
                    # zero stability is undefined, not evidence of a clean
                    # valley.  Some backbones/relations are then spuriously
                    # stable in a collapsed high-noise state before entering
                    # a genuine low-readability valley.  Anchor the monotone
                    # fit at that valley and detect recovery toward clean.
                    start = 0
                    if valley_anchored and len(values) > 2:
                        start = int(np.argmin(values[1:])) + 1
                        if start >= len(values) - 1:
                            start = 0
                    fitted = fitted_readability(values[start:])
                    slopes = np.diff(fitted) / delta_coordinate[start:]
                    if len(slopes) == 0:
                        return min(start, len(values) - 1), 0.0
                    local_index = int(np.argmax(slopes))
                    return start + local_index + 1, float(slopes[local_index])

                if edge_specific:
                    # Different semantic relations can become readable at
                    # different noise levels.  Select an independent knee for
                    # every typed edge, then assemble one routed graph whose
                    # edges originate from their own best trajectory states.
                    edge_ids = sorted(current_program_signals)
                    for edge_id in edge_ids:
                        fractions = np.asarray([
                            item["masks"].get(edge_id, np.zeros(samples, dtype=bool)).mean()
                            for item in route_candidates
                        ], dtype=float)
                        selected_index, selected_slope = select_transition(fractions)
                        selected = route_candidates[selected_index]
                        routed_program_signals[edge_id] = selected["signals"][edge_id]
                        routed_program_masks[edge_id] = selected["masks"][edge_id]
                        selected_coordinates[edge_id] = float(selected["coordinate"])
                        selected_slopes[edge_id] = selected_slope
                else:
                    fractions = np.asarray(
                        [item["stable_fraction"] for item in route_candidates], dtype=float)
                    selected_index, selected_slope = select_transition(fractions)
                    selected = route_candidates[selected_index]
                    routed_program_signals = selected["signals"]
                    routed_program_masks = selected["masks"]
                    selected_coordinates["global"] = float(selected["coordinate"])
                    selected_slopes["global"] = selected_slope
                route_captured = True
                active_counts = [int(value.sum()) for value in routed_program_masks.values()]
                step["projection_route_captured"] = True
                step["projection_route_mode"] = projection_route_mode
                step["projection_auto_route_coordinates"] = selected_coordinates
                step["projection_auto_max_stability_slopes"] = selected_slopes
                step["projection_routed_row_edges"] = int(sum(active_counts))
                step["projection_routed_row_fraction"] = float(
                    sum(active_counts) / max(samples * len(active_counts), 1))
        elif (projection_route_mode == "fixed" and projection_route_t >= 0
                and not route_captured
                and projection_coordinate <= projection_route_t):
            for edge_id, (signal, scale) in current_program_signals.items():
                routed_program_signals[edge_id] = signal.copy()
                routed_program_masks[edge_id] = current_stable_masks[edge_id].copy()
            route_captured = True
            counts = [int(value.sum()) for value in routed_program_masks.values()]
            step["projection_route_captured"] = True
            step["projection_routed_row_edges"] = int(sum(counts))
            step["projection_routed_row_fraction"] = float(
                sum(counts) / max(samples * len(counts), 1))
        effective_guidance_stage = "both" if guide_second_order else guidance_stage
        # The corrector-only schedule deliberately skips the predictor action.
        # Keep these inputs defined so the Heun correction can reuse the same
        # routed/stable graph state without depending on predictor execution.
        edge_row_masks = None
        edge_signal_overrides = None
        if projection_route_mode in automatic_route_modes or projection_route_t >= 0:
            edge_row_masks = routed_program_masks
            edge_signal_overrides = routed_program_signals
        elif projection_stability_threshold > 0:
            edge_row_masks = current_stable_masks
        if benefit_gate is not None and projection_weight > 0 and projection_strength > 0:
            benefit_masks, benefit_diagnostics = benefit_gate.row_masks(
                frame, denoised, sigma_num[index], t, factors)
            if edge_row_masks is None:
                edge_row_masks = benefit_masks
            else:
                edge_row_masks = dict(edge_row_masks)
                for edge_id, benefit_mask in benefit_masks.items():
                    edge_row_masks[edge_id] = (
                        edge_row_masks.get(edge_id, np.ones(samples, dtype=bool))
                        & benefit_mask
                    )
            step.update(benefit_diagnostics)
        if (projection_weight > 0 and projection_strength > 0
                and effective_guidance_stage in {"predictor", "both"}):
            if projection_route_mode in automatic_route_modes or projection_route_t >= 0:
                active_counts = [int(value.sum()) for value in edge_row_masks.values()]
                step["projection_stable_row_edges"] = int(sum(active_counts))
                step["projection_stable_row_fraction"] = float(
                    sum(active_counts) / max(samples * len(active_counts), 1))
            elif projection_stability_threshold > 0:
                active_counts = [int(value.sum()) for value in edge_row_masks.values()]
                step["projection_stable_row_edges"] = int(sum(active_counts))
                step["projection_stable_row_fraction"] = float(
                    sum(active_counts) / max(samples * len(active_counts), 1))
            before_projection = corrected.copy()
            corrected, projection_mask, events = factors.project_violations(
                corrected, projection_strength * projection_weight
                if edge_weights is None else projection_strength,
                edge_weights=edge_weights, edge_row_masks=edge_row_masks,
                edge_signal_overrides=edge_signal_overrides)
            changed_rows = projection_mask.any(axis=1)
            rejected_rows = np.zeros(samples, dtype=bool)
            if projection_density_veto and changed_rows.any() and factors.models:
                selected = np.flatnonzero(changed_rows)
                energy_before = factors.row_energy(before_projection.iloc[selected])
                energy_after = factors.row_energy(corrected.iloc[selected])
                rejected_local = energy_after > energy_before + projection_density_slack
                rejected_rows[selected[rejected_local]] = True
                if rejected_rows.any():
                    corrected.loc[rejected_rows, factors.numeric_names] = before_projection.loc[
                        rejected_rows, factors.numeric_names].to_numpy()
                    projection_mask[rejected_rows] = False
            mask |= projection_mask
            step["projection_events"] = events
            step["projection_rows_proposed"] = int(changed_rows.sum())
            step["projection_rows_density_rejected"] = int(rejected_rows.sum())

        corrected_encoded = torch.as_tensor(
            factors.encode_numeric(corrected), device=device, dtype=denoised.dtype)
        correction_mask = torch.as_tensor(mask, device=device)
        corrected_clean = denoised.clone()
        correction_delta = corrected_encoded - denoised
        trust_clipped_rows = 0
        if projection_trust_radius > 0 and correction_mask.any():
            # DSG-inspired ellipsoidal trust region.  Noise is feature-specific
            # in TabDiff, so measure the proposed clean-state displacement in
            # units of the current learned sigma instead of raw table units.
            normalized = torch.where(
                correction_mask,
                correction_delta / sigma_num[index].clamp_min(1e-8),
                torch.zeros_like(correction_delta),
            )
            dimensions = correction_mask.sum(dim=1).clamp_min(1)
            norm = normalized.square().sum(dim=1).sqrt()
            radius = projection_trust_radius * dimensions.to(norm.dtype).sqrt()
            row_scale = torch.minimum(torch.ones_like(norm), radius / norm.clamp_min(1e-12))
            trust_clipped_rows = int((row_scale < 1 - 1e-7).sum())
            correction_delta = correction_delta * row_scale[:, None]
        corrected_clean[correction_mask] = (denoised + correction_delta)[correction_mask]
        step["trust_clipped_rows"] = trust_clipped_rows
        step["mean_encoded_correction_l2"] = float(
            torch.where(correction_mask, correction_delta, torch.zeros_like(correction_delta))
            .square().sum(dim=1).sqrt().mean()
        )
        direction = (numeric_hat - corrected_clean) / sigma_num_hat[index]
        numeric_next = numeric_hat + (sigma_num_next[index] - sigma_num_hat[index]) * direction

        logits = diffusion._subs_parameterization(raw_logits, categorical_hat)
        alpha_t = torch.exp(-sigma_cat_hat[index]).unsqueeze(0).repeat(samples, 1)
        alpha_s = torch.exp(-sigma_cat_next[index]).unsqueeze(0).repeat(samples, 1)
        categorical_next, _ = diffusion._mdlm_update(logits, categorical_hat, alpha_t, alpha_s)
        if second_order_correction and index > 0:
            # Heun corrector from the official sampler.  Graph actuation is
            # deliberately applied to the predictor only: the corrector is an
            # unguided manifold-restoration step, analogous to releasing
            # guidance after a bounded intervention.
            categorical_hat_oh = diffusion.to_one_hot(categorical_hat).to(numeric_next.dtype)
            denoised_prime, raw_logits_prime = preview.base(
                numeric_next.float(), categorical_hat_oh,
                times[index - 1].squeeze().repeat(samples),
                sigma=sigma_num_next[index].unsqueeze(0).repeat(samples, 1),
            )
            if (effective_guidance_stage in {"corrector", "both"}
                    and projection_weight > 0 and projection_strength > 0):
                prime_codes = preview.predicted_codes(
                    categorical_hat_oh, raw_logits_prime)
                prime_frame = decode(
                    train_data,
                    torch.cat([denoised_prime, prime_codes.float()], dim=1).cpu(), info,
                )
                prime_corrected, prime_mask, prime_events = factors.project_violations(
                    prime_frame,
                    projection_strength * projection_weight
                    if edge_weights is None else projection_strength,
                    edge_weights=edge_weights,
                    edge_row_masks=edge_row_masks,
                    edge_signal_overrides=edge_signal_overrides,
                )
                prime_encoded = torch.as_tensor(
                    factors.encode_numeric(prime_corrected),
                    device=device, dtype=denoised_prime.dtype,
                )
                prime_mask_tensor = torch.as_tensor(prime_mask, device=device)
                prime_delta = prime_encoded - denoised_prime
                if projection_trust_radius > 0 and prime_mask_tensor.any():
                    normalized_prime = torch.where(
                        prime_mask_tensor,
                        prime_delta / sigma_num_next[index].clamp_min(1e-8),
                        torch.zeros_like(prime_delta),
                    )
                    prime_dimensions = prime_mask_tensor.sum(dim=1).clamp_min(1)
                    prime_norm = normalized_prime.square().sum(dim=1).sqrt()
                    prime_radius = projection_trust_radius * prime_dimensions.to(
                        prime_norm.dtype).sqrt()
                    prime_scale = torch.minimum(
                        torch.ones_like(prime_norm),
                        prime_radius / prime_norm.clamp_min(1e-12),
                    )
                    prime_delta = prime_delta * prime_scale[:, None]
                denoised_prime = denoised_prime.clone()
                denoised_prime[prime_mask_tensor] = (
                    denoised_prime + prime_delta)[prime_mask_tensor]
                step["second_order_projection_events"] = int(prime_events)
                step["projection_events"] += int(prime_events)
            direction_prime = (numeric_next - denoised_prime) / sigma_num_next[index]
            numeric_next = numeric_hat + (
                sigma_num_next[index] - sigma_num_hat[index]
            ) * (0.5 * direction + 0.5 * direction_prime)
        step["stochastic_sampler"] = bool(stochastic_sampler)
        step["second_order_correction"] = bool(second_order_correction)
        step["guidance_stage"] = effective_guidance_stage
        step["projection_persistence_steps"] = int(projection_persistence_steps)
        numeric, categorical = numeric_next.detach(), categorical_next.detach()
        previous_program_signals = {
            edge_id: signal.copy()
            for edge_id, (signal, _) in current_program_signals.items()
        }
        for key in ("factor_cell_events", "projection_events", "standardized_movement",
                    "transport_cell_events", "transport_row_events",
                    "categorical_guidance_events",
                    "factor_candidate_row_factors", "factor_active_row_factors",
                    "factor_low_residual_rejections", "factor_uncertainty_rejections",
                    "factor_satisfied_candidate_row_factors",
                    "factor_violating_candidate_row_factors",
                    "factor_satisfied_active_row_factors",
                    "factor_violating_active_row_factors",
                    "factor_satisfied_cell_events", "factor_violating_cell_events",
                    "benefit_gate_rows_proposed", "benefit_gate_rows_accepted",
                    "transport_uncertain_row_events",
                    "transport_tail_rows_preserved"):
            step.setdefault(key, 0)
            totals[key] += float(step[key])
        for key in (
            "transport_posterior_entropy_mean",
            "transport_posterior_entropy_p90",
            "transport_posterior_max_probability_mean",
            "transport_bin_switch_fraction",
            "transport_membership_l1_drift",
            "transport_membership_hysteresis",
            "transport_switch_margin",
            "transport_switch_candidate_fraction",
            "transport_switch_held_fraction",
            "transport_confidence_floor",
            "transport_confidence_full",
            "transport_fallback_scale",
            "transport_confidence_gate_mean",
            "transport_confidence_gate_p10",
            "transport_tail_fraction_preserved",
            "transport_mean_absolute_move",
            "transport_local_mean_absolute_move",
            "transport_global_mean_absolute_move",
            "transport_pre_reserve_mean_absolute_move",
            "transport_standardized_move_p90",
            "transport_bin_0_posterior_mass",
            "transport_bin_1_posterior_mass",
            "transport_bin_2_posterior_mass",
            "transport_bin_3_posterior_mass",
            "transport_bin_0_primary_fraction",
            "transport_bin_1_primary_fraction",
            "transport_bin_2_primary_fraction",
            "transport_bin_3_primary_fraction",
        ):
            if key in step:
                totals[key] += float(step[key])
                totals[f"{key}_observations"] += 1.0
        trace.append(step)

    if not torch.all(categorical < diffusion.mask_index):
        raise RuntimeError("categorical mask remained at final step")
    totals["factor_active_fraction"] = (
        totals["factor_active_row_factors"] / totals["factor_candidate_row_factors"]
        if totals["factor_candidate_row_factors"] else 0.0)
    totals["transport_calls"] = int(transport_calls)
    for key in (
        "transport_posterior_entropy_mean",
        "transport_posterior_entropy_p90",
        "transport_posterior_max_probability_mean",
        "transport_bin_switch_fraction",
        "transport_membership_l1_drift",
        "transport_membership_hysteresis",
        "transport_switch_margin",
        "transport_switch_candidate_fraction",
        "transport_switch_held_fraction",
        "transport_confidence_floor",
        "transport_confidence_full",
        "transport_fallback_scale",
        "transport_confidence_gate_mean",
        "transport_confidence_gate_p10",
        "transport_tail_fraction_preserved",
        "transport_mean_absolute_move",
        "transport_local_mean_absolute_move",
        "transport_global_mean_absolute_move",
        "transport_pre_reserve_mean_absolute_move",
        "transport_standardized_move_p90",
        "transport_bin_0_posterior_mass",
        "transport_bin_1_posterior_mass",
        "transport_bin_2_posterior_mass",
        "transport_bin_3_posterior_mass",
        "transport_bin_0_primary_fraction",
        "transport_bin_1_primary_fraction",
        "transport_bin_2_primary_fraction",
        "transport_bin_3_primary_fraction",
    ):
        observations = totals.pop(f"{key}_observations", 0.0)
        if observations:
            totals[key] /= observations
    return torch.cat([numeric, categorical.float()], dim=1).cpu(), {
        **{key: (int(value) if key.endswith("events") else value) for key, value in totals.items()},
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="news")
    parser.add_argument(
        "--regression-target-mode", choices=["categorical", "numerical"], default="categorical",
        help="Use numerical to match released TabDiff preprocessing for regression datasets.",
    )
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--include-families", default="",
                        help="Optional comma-separated relation_family allow-list.")
    parser.add_argument("--include-edge-ids", default="",
                        help="Optional comma-separated candidate_id allow-list, calibrated on train only.")
    parser.add_argument("--family-strengths", default="",
                        help="Optional relation_family=multiplier comma-separated mapping.")
    parser.add_argument("--program-transport", choices=["direct", "rank"], default="direct")
    parser.add_argument("--family-transports", default="",
                        help="Optional relation_family=direct|rank mapping overriding --program-transport.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261601)
    parser.add_argument("--mode", choices=["residual", "copula"], default="copula")
    parser.add_argument("--strength", type=float, default=.25)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--min-r2", type=float, default=.02)
    parser.add_argument("--factor-usage-roles", default="",
                        help="Optional comma-separated usage_role allow-list for train-fitted factor messages. Projection still sees the full graph.")
    parser.add_argument("--factor-families", default="",
                        help="Optional relation_family allow-list for factor messages. Projection still sees the full graph.")
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--factor-min-t", type=float, default=.15)
    parser.add_argument("--factor-max-t", type=float, default=.65)
    parser.add_argument("--factor-schedule", choices=["flat", "late_ramp", "middle_peak"], default="middle_peak")
    parser.add_argument("--factor-residual-gate", type=float, default=0.0,
                        help="Activate a row-factor only above this normalized conditional residual; 0 disables.")
    parser.add_argument("--factor-uncertainty-gate", type=float, default=-1.0,
                        help="Maximum validation-q90-normalized ExtraTrees disagreement; negative disables.")
    parser.add_argument("--factor-uncertainty-trees", type=int, default=16,
                        help="Number of trees used for the runtime uncertainty proxy.")
    parser.add_argument("--factor-satisfied-strength-scale", type=float, default=1.0,
                        help="Relative message weight for applicable edges currently satisfied.")
    parser.add_argument("--factor-violating-strength-scale", type=float, default=1.0,
                        help="Relative message weight for applicable edges currently violated.")
    parser.add_argument("--interval-space", choices=["time", "log_sigma_quantile"], default="time")
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--projection-min-t", type=float, default=0.0)
    parser.add_argument("--projection-max-t", type=float, default=.50)
    parser.add_argument("--projection-interval-space", choices=["time", "log_sigma_quantile", "edge_log_sigma_quantile"], default="time")
    parser.add_argument("--projection-schedule", choices=["flat", "late_ramp", "middle_peak"], default="flat")
    parser.add_argument("--projection-trust-radius", type=float, default=0.0,
                        help="If positive, cap each row correction in feature-sigma units; 0 disables the trust region.")
    parser.add_argument("--projection-density-veto", action="store_true",
                        help="Reject row-level graph corrections that worsen train-fitted conditional energy.")
    parser.add_argument("--projection-density-slack", type=float, default=0.0)
    parser.add_argument("--projection-stability-threshold", type=float, default=0.0,
                        help="If positive, actuate a program only on rows whose proposed target is stable across consecutive reverse states under the selected mode.")
    parser.add_argument("--projection-stability-mode", choices=["scale", "rank"], default="scale",
                        help="Measure consecutive-state stability in train-scale units or normalized within-batch rank displacement.")
    parser.add_argument("--projection-route-t", type=float, default=-1.0,
                        help="If nonnegative, freeze the row mask and program target when the projection coordinate first crosses this middle-state value; execute only in the ordinary projection window.")
    parser.add_argument("--projection-route-mode", choices=[
                            "fixed", "auto_knee", "auto_edge_knee",
                            "auto_isotonic", "auto_edge_isotonic",
                            "auto_valley_isotonic", "auto_edge_valley_isotonic"], default="fixed",
                        help="fixed uses --projection-route-t; auto_knee uses the raw cumulative stability curve; auto_isotonic first pools transient reversals; edge variants select a state independently for every typed edge.")
    parser.add_argument("--projection-benefit-gate", type=Path,
                        help="Optional train-only benefit-gate manifest for row/edge action selection.")
    parser.add_argument("--benefit-gate-edge-ids", default="",
                        help="Optional comma-separated subset gated; other graph edges remain unchanged.")
    parser.add_argument("--stochastic-sampler", action="store_true",
                        help="Use the released TabDiff EDM churn/noise update for baseline and guided paths.")
    parser.add_argument("--second-order-correction", action="store_true",
                        help="Use the released TabDiff Heun corrector; graph guidance acts on the predictor only.")
    parser.add_argument("--guide-second-order", action="store_true",
                        help="Apply the routed typed projection at both Heun predictor and corrector evaluations.")
    parser.add_argument("--guidance-stage", choices=["predictor", "corrector", "both"],
                        default="predictor",
                        help="Inject the typed action in the Heun predictor, corrector, or both.")
    parser.add_argument("--projection-persistence-steps", type=int, default=1,
                        help="Require row-edge stability for this many consecutive reverse states before activation.")
    parser.add_argument("--distribution-transport-mode",
                        choices=["none", "marginal", "copula", "pearson_copula",
                                 "graph_copula", "adaptive_copula", "sliced",
                                 "graph_sliced", "adaptive_sliced"],
                        default="none",
                        help="Optional train-only batch distribution correction on clean previews.")
    parser.add_argument("--distribution-transport-scope", choices=["graph", "all"],
                        default="graph")
    parser.add_argument("--distribution-transport-strength", type=float, default=0.0)
    parser.add_argument("--distribution-transport-min-t", type=float, default=.20,
                        help="Lower log-sigma quantile for batch transport.")
    parser.add_argument("--distribution-transport-max-t", type=float, default=.35,
                        help="Upper log-sigma quantile for batch transport.")
    parser.add_argument("--distribution-transport-schedule",
                        choices=["flat", "late_ramp", "middle_peak"], default="flat")
    parser.add_argument("--distribution-transport-top-k", type=int, default=0,
                        help="For adaptive_copula, activate at most this many largest state-specific pair gaps; 0 keeps every gap above threshold.")
    parser.add_argument("--distribution-transport-min-gap", type=float, default=0.0,
                        help="Minimum absolute train-preview Gaussian-rank correlation gap for adaptive activation.")
    parser.add_argument("--distribution-transport-sliced-directions", type=int, default=32,
                        help="Number of fixed candidate projections for sliced transport.")
    parser.add_argument("--distribution-transport-once", action="store_true",
                        help="Apply batch transport only at the first active middle state.")
    parser.add_argument("--distribution-transport-max-calls", type=int, default=0,
                        help="Maximum separated transport pulses; 0 leaves the active-window schedule unconstrained.")
    parser.add_argument("--distribution-transport-min-call-gap", type=float, default=0.0,
                        help="Minimum log-sigma-quantile distance between transport pulses.")
    parser.add_argument("--distribution-transport-condition-target", action="store_true",
                        help="Fit a separate numerical transport per categorical target value; subgroup membership follows the current clean preview.")
    parser.add_argument("--distribution-transport-condition-target-quantiles", type=int, default=0,
                        help="For regression, condition transport on this many train-fitted target quantile bins.")
    parser.add_argument("--distribution-transport-soft-quantiles", action="store_true",
                        help="Marginalize the two nearest target-quantile operators per row instead of hard binning.")
    parser.add_argument("--distribution-transport-soft-temperature", type=float, default=.75,
                        help="Positive posterior temperature for soft target-quantile membership.")
    parser.add_argument("--distribution-transport-freeze-membership", action="store_true",
                        help="Freeze row-specific soft membership after its first active pulse.")
    parser.add_argument(
        "--distribution-transport-confidence-floor", type=float, default=0.0,
        help="Posterior confidence below which the conditional action is attenuated.")
    parser.add_argument(
        "--distribution-transport-confidence-full", type=float, default=1.0,
        help="Posterior confidence at which the full conditional action is used.")
    parser.add_argument(
        "--distribution-transport-uncertainty-fallback",
        choices=["none", "global", "abstain"], default="none",
        help="For uncertain rows, blend toward global transport or abstain.")
    parser.add_argument(
        "--distribution-transport-fallback-scale", type=float, default=1.0,
        help="Strength multiplier for the global uncertainty fallback.")
    parser.add_argument(
        "--distribution-transport-membership-hysteresis", type=float, default=0.0,
        help="Previous-pulse posterior weight in [0,1) for dynamic routing stability.")
    parser.add_argument(
        "--distribution-transport-switch-margin", type=float, default=0.0,
        help="Minimum posterior advantage required to change primary bins.")
    parser.add_argument(
        "--distribution-transport-preserve-condition-column", action="store_true",
        help="Treat the target used for routing as read-only during transport.")
    parser.add_argument("--distribution-transport-preserve-tail", type=float, default=0.0,
                        help="Protect this high-rarity row fraction from batch transport to retain coverage.")
    parser.add_argument("--categorical-guidance-mode",
                        choices=["none", "marginal", "target", "pairwise",
                                 "marginal_target", "marginal_pairwise"],
                        default="none",
                        help="Train-only log-density messages applied to categorical clean logits.")
    parser.add_argument("--categorical-guidance-strength", type=float, default=0.0)
    parser.add_argument("--categorical-guidance-min-t", type=float, default=.20,
                        help="Lower log-sigma quantile for categorical logit guidance.")
    parser.add_argument("--categorical-guidance-max-t", type=float, default=.50,
                        help="Upper log-sigma quantile for categorical logit guidance.")
    parser.add_argument("--categorical-guidance-schedule",
                        choices=["flat", "late_ramp", "middle_peak"], default="flat")
    parser.add_argument("--categorical-guidance-max-pairs", type=int, default=16)
    parser.add_argument("--categorical-guidance-min-nmi", type=float, default=.01)
    parser.add_argument("--categorical-guidance-smoothing", type=float, default=.5)
    parser.add_argument("--categorical-guidance-max-joint-cells", type=int, default=4096)
    parser.add_argument("--categorical-guidance-clip", type=float, default=3.0)
    parser.add_argument("--categorical-guidance-marginal-scale", type=float, default=1.0)
    parser.add_argument("--categorical-guidance-pair-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.projection_persistence_steps < 1:
        parser.error("--projection-persistence-steps must be at least 1")
    if (args.distribution_transport_condition_target
            and args.distribution_transport_condition_target_quantiles > 1):
        parser.error("choose categorical or quantile target conditioning, not both")
    if ((args.guidance_stage in {"corrector", "both"} or args.guide_second_order)
            and not args.second_order_correction):
        parser.error("corrector guidance requires --second-order-correction")
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    started = time.perf_counter()
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, test_data = preprocess_dataset(
        args.repo, args.dataset, regression_target_mode=args.regression_target_mode)
    train = decode(train_data, train_data.X, info)
    heldout = decode(train_data, test_data.X, info)
    edges = json.loads(args.graph.read_text())
    if args.include_edge_ids:
        included_ids = {
            value.strip() for value in args.include_edge_ids.split(",") if value.strip()
        }
        edges = [edge for edge in edges if edge.get("candidate_id") in included_ids]
        if not edges:
            raise ValueError(f"no graph edges match --include-edge-ids={sorted(included_ids)}")
    if args.include_families:
        included = {value.strip() for value in args.include_families.split(",") if value.strip()}
        edges = [edge for edge in edges if edge.get("relation_family") in included]
        if not edges:
            raise ValueError(f"no graph edges match --include-families={sorted(included)}")
    if args.family_strengths:
        family_weights = {}
        for item in args.family_strengths.split(","):
            family, value = item.split("=", 1)
            family_weights[family.strip()] = float(value)
        for edge in edges:
            edge["guidance_weight"] = family_weights.get(edge.get("relation_family"), 0.0)
    if args.program_transport != "direct":
        for edge in edges:
            if edge.get("program"):
                edge["program"]["transport"] = args.program_transport
    if args.family_transports:
        family_transports = {}
        for item in args.family_transports.split(","):
            family, value = item.split("=", 1)
            value = value.strip()
            if value not in {"direct", "rank"}:
                raise ValueError(f"unsupported family transport: {value}")
            family_transports[family.strip()] = value
        for edge in edges:
            if edge.get("program") and edge.get("relation_family") in family_transports:
                edge["program"]["transport"] = family_transports[edge["relation_family"]]
    # Functional programs have their own typed semantics and are deliberately
    # absent from the legacy predicate-violation metric.  Main quality is
    # measured by the independent common evaluator; predicate violation is
    # retained only for graphs that actually contain legacy requirements.
    metric_edges = [edge for edge in edges if edge.get("requirements")]
    factor_usage_roles = {
        value.strip() for value in args.factor_usage_roles.split(",") if value.strip()
    }
    factor_relation_families = {
        value.strip() for value in args.factor_families.split(",") if value.strip()
    }
    factors = DenseNumericFactors(
        train_data, train, info, edges, args.seed, args.trees, args.min_r2,
        factor_usage_roles=factor_usage_roles or None,
        factor_relation_families=factor_relation_families or None,
    )
    batch_transport = None
    if args.distribution_transport_mode != "none":
        transport_kwargs = dict(
            scope=args.distribution_transport_scope,
            mode=args.distribution_transport_mode,
            adaptive_top_k=args.distribution_transport_top_k,
            adaptive_min_gap=args.distribution_transport_min_gap,
            preserve_tail_fraction=args.distribution_transport_preserve_tail,
            sliced_directions=args.distribution_transport_sliced_directions,
            seed=args.seed,
        )
        if args.distribution_transport_condition_target_quantiles > 1:
            if info["task_type"] != "regression":
                raise ValueError("quantile target conditioning requires regression")
            target_column = info["column_names"][info["target_col_idx"][0]]
            if args.distribution_transport_soft_quantiles:
                use_uncertainty_router = (
                    args.distribution_transport_uncertainty_fallback != "none"
                    or args.distribution_transport_membership_hysteresis > 0.0
                    or args.distribution_transport_switch_margin > 0.0
                    or args.distribution_transport_preserve_condition_column
                )
                router_class = (
                    UncertaintyAwarePosteriorQuantileTransport
                    if use_uncertainty_router
                    else PosteriorQuantileConditionalBatchDistributionTransport
                )
                router_kwargs = {}
                if use_uncertainty_router:
                    router_kwargs.update(
                        confidence_floor=(
                            args.distribution_transport_confidence_floor
                        ),
                        confidence_full=(
                            args.distribution_transport_confidence_full
                        ),
                        uncertainty_fallback=(
                            args.distribution_transport_uncertainty_fallback
                        ),
                        fallback_scale=(
                            args.distribution_transport_fallback_scale
                        ),
                        membership_hysteresis=(
                            args.distribution_transport_membership_hysteresis
                        ),
                        switch_margin=(
                            args.distribution_transport_switch_margin
                        ),
                        preserve_condition_column=(
                            args.distribution_transport_preserve_condition_column
                        ),
                    )
                batch_transport = router_class(
                    train, numeric_columns(info), edges,
                    condition_column=target_column,
                    quantile_bins=args.distribution_transport_condition_target_quantiles,
                    temperature=args.distribution_transport_soft_temperature,
                    freeze_membership=args.distribution_transport_freeze_membership,
                    **router_kwargs, **transport_kwargs)
            else:
                batch_transport = QuantileConditionalBatchDistributionTransport(
                    train, numeric_columns(info), edges,
                    condition_column=target_column,
                    quantile_bins=args.distribution_transport_condition_target_quantiles,
                    **transport_kwargs)
        elif args.distribution_transport_condition_target:
            if info["task_type"] == "regression":
                raise ValueError("categorical target conditioning is unavailable for regression")
            target_column = info["column_names"][info["target_col_idx"][0]]
            batch_transport = ConditionalBatchDistributionTransport(
                train, numeric_columns(info), edges,
                condition_column=target_column, **transport_kwargs)
        else:
            batch_transport = BatchDistributionTransport(
                train, numeric_columns(info), edges, **transport_kwargs)
    categorical_guidance = None
    if args.categorical_guidance_mode != "none":
        categorical_guidance = CategoricalLogitGuidance(
            train_data, info, mode=args.categorical_guidance_mode,
            max_pairs=args.categorical_guidance_max_pairs,
            min_nmi=args.categorical_guidance_min_nmi,
            smoothing=args.categorical_guidance_smoothing,
            max_joint_cells=args.categorical_guidance_max_joint_cells,
            message_clip=args.categorical_guidance_clip,
            marginal_scale=args.categorical_guidance_marginal_scale,
            pair_scale=args.categorical_guidance_pair_scale,
        )
    benefit_gate_ids = {
        value.strip() for value in args.benefit_gate_edge_ids.split(",") if value.strip()
    }
    benefit_gate = (
        BenefitGate(args.projection_benefit_gate, benefit_gate_ids or None)
        if args.projection_benefit_gate is not None else None
    )
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["denoise_fn"], strict=True)
    preview = PreviewModel(model, train_data.categories)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column",
                            cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(
        stochastic_sampler=args.stochastic_sampler,
        second_order_correction=args.second_order_correction,
    )
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, preview, None,
        **diffusion_params, device=torch.device("cuda"),
    ).cuda()
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()

    sample_dir = args.output / "samples"; sample_dir.mkdir()
    common_seed = args.seed + 100_000
    seed_all(common_seed)
    unconditional = decode(train_data, diffusion.sample(args.samples), info)
    unconditional.to_csv(sample_dir / "unconditional.csv", index=False)
    seed_all(common_seed)
    encoded, diagnostics = sample(
        diffusion, preview, train_data, info, factors, args.samples,
        args.mode, args.strength, args.passes, args.clip,
        args.factor_residual_gate, args.factor_uncertainty_gate,
        args.factor_uncertainty_trees,
        args.factor_satisfied_strength_scale,
        args.factor_violating_strength_scale,
        args.factor_min_t, args.factor_max_t, args.factor_schedule,
        args.interval_space, args.projection_strength, args.projection_min_t,
        args.projection_max_t,
        args.projection_interval_space, args.projection_schedule,
        args.projection_trust_radius, args.projection_density_veto,
        args.projection_density_slack, args.projection_stability_threshold,
        args.projection_stability_mode, args.projection_route_t,
        args.projection_route_mode, benefit_gate, args.stochastic_sampler,
        args.second_order_correction, args.guide_second_order,
        args.guidance_stage, args.projection_persistence_steps,
        batch_transport, args.distribution_transport_strength,
        args.distribution_transport_min_t, args.distribution_transport_max_t,
        args.distribution_transport_schedule, args.distribution_transport_once,
        args.distribution_transport_max_calls,
        args.distribution_transport_min_call_gap,
        categorical_guidance, args.categorical_guidance_strength,
        args.categorical_guidance_min_t, args.categorical_guidance_max_t,
        args.categorical_guidance_schedule,
    )
    guided = decode(train_data, encoded, info)
    guided.to_csv(sample_dir / "dense_factor.csv", index=False)
    repaired, repair_diagnostics = factors.certify(guided)
    # Program projection promotes numerical columns to float64.  Cast them
    # back before serialization: otherwise 1e-8 representation differences
    # split large groups of tied values and create a spurious KS/Shape loss.
    repaired = repaired.astype({column: guided[column].dtype for column in guided.columns})
    repaired.to_csv(sample_dir / "dense_factor_then_certifier.csv", index=False)
    base_repaired, base_repair_diagnostics = factors.certify(unconditional)
    base_repaired = base_repaired.astype(
        {column: unconditional[column].dtype for column in unconditional.columns})
    base_repaired.to_csv(sample_dir / "final_repair.csv", index=False)

    rows = [
        {"method": "unconditional", **graph_metrics(unconditional, heldout, metric_edges)},
        {"method": "final_repair", **graph_metrics(base_repaired, heldout, metric_edges),
         "repair_events": base_repair_diagnostics["action_events"]},
        {"method": "dense_factor", **graph_metrics(guided, heldout, metric_edges), **diagnostics},
        {"method": "dense_factor_then_certifier", **graph_metrics(repaired, heldout, metric_edges),
         **diagnostics, "repair_events": repair_diagnostics["action_events"]},
    ]
    pd.DataFrame(rows).drop(columns=["trace"], errors="ignore").to_csv(args.output / "mechanism_metrics.csv", index=False)
    (args.output / "trace.json").write_text(json.dumps(diagnostics, indent=2))
    metadata = {
        **vars(args), "repo": str(args.repo), "graph": str(args.graph),
        "config": str(args.config), "checkpoint": str(args.checkpoint), "output": str(args.output),
        "factors": factors.metadata(), "factor_count": len(factors.models),
        "distribution_transport_columns": (
            batch_transport.columns if batch_transport is not None else []),
        "categorical_guidance": (
            categorical_guidance.metadata() if categorical_guidance is not None else None),
        "runtime_seconds": time.perf_counter() - started,
        "mechanism": "dense numeric factors with optional copula and categorical log-density guidance",
        "heldout_used_for_fitting": False, "api_calls": 0,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    printable = pd.DataFrame(rows).drop(columns=["trace"], errors="ignore")
    lines = ["# Dense numerical factor messages", "", f"- Factors retained: **{len(factors.models)}**",
             f"- Mode: `{args.mode}`; strength `{args.strength}`; passes `{args.passes}`", "",
             "```text", printable.to_string(index=False), "```"]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(pd.DataFrame(rows).drop(columns=["trace"], errors="ignore").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
