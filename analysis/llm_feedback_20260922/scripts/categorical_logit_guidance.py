#!/usr/bin/env python3
"""Train-only categorical density messages for masked tabular diffusion.

TabDiff predicts a clean categorical distribution immediately before every
MDLM unmasking transition.  This module adds bounded log-density-ratio
messages to those clean logits.  It never changes an already unmasked value
and never reads validation/test rows.

Two complementary potentials are available:

``marginal``
    Match the expected batch marginal to the empirical training marginal.
    At each state, subtract probability mass already committed by absorbing
    unmasking and balance only the residual mass of cells that can still move.
``target`` / ``pairwise``
    Send positive *and* negative information over train-fitted categorical
    factors.  A message is the expectation of a smoothed pointwise mutual
    information table under the other node's current clean posterior.

The latter differs from violation repair: every readable state sends a soft
message, including states that already satisfy the likely relation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CategoricalPair:
    left: int
    right: int
    normalized_mi: float
    pmi: torch.Tensor


def _entropy(probability: np.ndarray) -> float:
    positive = probability > 0
    return float(-(probability[positive] * np.log(probability[positive])).sum())


class CategoricalLogitGuidance:
    """Categorical marginal and dependency potentials fitted on training rows."""

    MODES = {
        "marginal", "target", "pairwise", "marginal_target",
        "marginal_pairwise",
    }

    def __init__(
        self,
        train_data,
        info: dict,
        mode: str = "marginal_target",
        max_pairs: int = 16,
        min_nmi: float = 0.01,
        smoothing: float = 0.5,
        max_joint_cells: int = 4096,
        message_clip: float = 3.0,
        marginal_scale: float = 1.0,
        pair_scale: float = 1.0,
    ):
        if mode not in self.MODES:
            raise ValueError(f"unsupported categorical guidance mode: {mode}")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        if message_clip <= 0:
            raise ValueError("message_clip must be positive")
        if marginal_scale < 0 or pair_scale < 0:
            raise ValueError("categorical guidance component scales must be nonnegative")
        self.mode = mode
        self.category_sizes = [int(value) for value in train_data.categories]
        self.message_clip = float(message_clip)
        self.marginal_scale = float(marginal_scale)
        self.pair_scale = float(pair_scale)
        self.use_marginal = mode.startswith("marginal")
        self.use_pairs = mode in {
            "target", "pairwise", "marginal_target", "marginal_pairwise",
        }

        feature_names = [info["column_names"][i] for i in info["cat_col_idx"]]
        target_names = [info["column_names"][i] for i in info["target_col_idx"]]
        regression_target_is_numerical = (
            info["task_type"] == "regression"
            and train_data.d_numerical == len(info["num_col_idx"]) + len(target_names)
        )
        self.target_position = None if regression_target_is_numerical else 0
        self.column_names = (
            feature_names if regression_target_is_numerical
            else target_names + feature_names
        )
        if len(self.column_names) != len(self.category_sizes):
            raise ValueError(
                "categorical column layout disagrees with fitted transformer: "
                f"{len(self.column_names)} names vs {len(self.category_sizes)} sizes"
            )
        if "target" in mode and self.target_position is None:
            raise ValueError("target-posterior categorical guidance requires a categorical target")

        encoded = train_data.X[:, train_data.d_numerical:].detach().cpu().numpy().astype(int)
        if encoded.shape[1] != len(self.category_sizes):
            raise ValueError("encoded categorical width disagrees with category sizes")
        n = len(encoded)
        self.marginals: list[torch.Tensor] = []
        empirical_marginals: list[np.ndarray] = []
        for index, size in enumerate(self.category_sizes):
            count = np.bincount(encoded[:, index], minlength=size).astype(np.float64)
            probability = (count + smoothing) / (n + smoothing * size)
            empirical_marginals.append(probability)
            self.marginals.append(torch.from_numpy(probability.astype(np.float32)))

        if not self.use_pairs:
            candidate_indices: list[tuple[int, int]] = []
        elif "target" in mode:
            candidate_indices = [
                (int(self.target_position), index)
                for index in range(len(self.category_sizes))
                if index != self.target_position
            ]
        else:
            candidate_indices = list(combinations(range(len(self.category_sizes)), 2))

        candidates: list[CategoricalPair] = []
        for left, right in candidate_indices:
            left_size, right_size = self.category_sizes[left], self.category_sizes[right]
            if left_size * right_size > max_joint_cells:
                continue
            flat = encoded[:, left] * right_size + encoded[:, right]
            count = np.bincount(flat, minlength=left_size * right_size).reshape(
                left_size, right_size).astype(np.float64)
            empirical_joint = count / max(count.sum(), 1.0)
            left_empirical = empirical_joint.sum(axis=1)
            right_empirical = empirical_joint.sum(axis=0)
            independent = np.outer(left_empirical, right_empirical)
            positive = empirical_joint > 0
            mutual_information = float((empirical_joint[positive] * np.log(
                empirical_joint[positive] / np.maximum(independent[positive], 1e-12)
            )).sum())
            denominator = max(
                np.sqrt(_entropy(left_empirical) * _entropy(right_empirical)), 1e-12)
            normalized_mi = mutual_information / denominator
            if not np.isfinite(normalized_mi) or normalized_mi < min_nmi:
                continue

            # A symmetric Dirichlet prior keeps rare category messages finite.
            joint = (count + smoothing) / (
                count.sum() + smoothing * left_size * right_size)
            left_probability = joint.sum(axis=1)
            right_probability = joint.sum(axis=0)
            pmi = np.log(joint) - np.log(left_probability[:, None]) - np.log(
                right_probability[None, :])
            candidates.append(CategoricalPair(
                left=left,
                right=right,
                normalized_mi=float(np.clip(normalized_mi, 0.0, 1.0)),
                pmi=torch.from_numpy(pmi.astype(np.float32)),
            ))
        candidates.sort(key=lambda pair: pair.normalized_mi, reverse=True)
        self.pairs = candidates[:max_pairs] if max_pairs > 0 else candidates

    def metadata(self) -> dict:
        return {
            "mode": self.mode,
            "columns": self.column_names,
            "target_position": self.target_position,
            "pair_count": len(self.pairs),
            "marginal_scale": self.marginal_scale,
            "pair_scale": self.pair_scale,
            "pairs": [
                {
                    "left": self.column_names[pair.left],
                    "right": self.column_names[pair.right],
                    "normalized_mi": pair.normalized_mi,
                }
                for pair in self.pairs
            ],
        }

    def apply(
        self,
        raw_logits: torch.Tensor,
        current_codes: torch.Tensor,
        weight: float,
    ) -> tuple[torch.Tensor, dict]:
        """Guide logits for cells that are still masked.

        ``current_codes`` uses TabDiff's absorbing mask code ``K_j``.  The
        expected batch distribution includes one-hot mass from already
        unmasked cells and clean posterior mass from masked cells.
        """
        if weight <= 0:
            return raw_logits, {
                "categorical_guidance_events": 0,
                "categorical_guidance_masked_fraction": 0.0,
                "categorical_guidance_mean_logit_shift": 0.0,
                "categorical_guidance_tv_before": 0.0,
                "categorical_guidance_tv_after": 0.0,
            }
        chunks = list(torch.split(
            raw_logits, [size + 1 for size in self.category_sizes], dim=-1))
        if len(chunks) != len(self.category_sizes):
            raise ValueError("raw categorical logits have an unexpected layout")

        beliefs: list[torch.Tensor] = []
        masked: list[torch.Tensor] = []
        for index, (chunk, size) in enumerate(zip(chunks, self.category_sizes)):
            posterior = F.softmax(chunk[:, :size], dim=-1)
            is_masked = current_codes[:, index].eq(size)
            safe_code = current_codes[:, index].clamp(max=size - 1)
            observed = F.one_hot(safe_code, num_classes=size).to(posterior.dtype)
            beliefs.append(torch.where(is_masked[:, None], posterior, observed))
            masked.append(is_masked)

        messages = [torch.zeros_like(belief) for belief in beliefs]
        normalizers = [0.0 for _ in beliefs]
        if self.use_pairs:
            for pair in self.pairs:
                pmi = pair.pmi.to(device=raw_logits.device, dtype=raw_logits.dtype)
                reliability = pair.normalized_mi
                messages[pair.left] += reliability * beliefs[pair.right] @ pmi.T
                messages[pair.right] += reliability * beliefs[pair.left] @ pmi
                normalizers[pair.left] += reliability
                normalizers[pair.right] += reliability

        before_tv = []
        after_tv = []
        shifts = []
        guided_beliefs = []
        guided_chunks = []
        event_count = 0
        for index, (chunk, belief, is_masked) in enumerate(
                zip(chunks, beliefs, masked)):
            target = self.marginals[index].to(
                device=raw_logits.device, dtype=raw_logits.dtype)
            message = messages[index]
            if normalizers[index] > 0:
                message = message / normalizers[index]
                message = message - (belief * message).sum(dim=1, keepdim=True)
                message = self.pair_scale * message
            if self.use_marginal:
                # MDLM's unmasking is absorbing.  Solve the marginal target for
                # the probability mass that is *still movable*, rather than
                # asking frozen cells to compensate for an existing imbalance.
                masked_count = int(is_masked.sum())
                if masked_count:
                    frozen_mass = belief[~is_masked].sum(dim=0)
                    residual_target = (
                        len(belief) * target - frozen_mass).clamp_min(1e-7)
                    residual_target = residual_target / residual_target.sum()
                    current_movable = belief[is_masked].mean(dim=0).clamp_min(1e-7)
                    marginal_vector = (
                        torch.log(residual_target) - torch.log(current_movable))
                else:
                    marginal_vector = torch.zeros_like(target)
                marginal_message = marginal_vector[None, :].expand_as(message)
                marginal_message = marginal_message - (
                    belief * marginal_message).sum(dim=1, keepdim=True)
                message = message + self.marginal_scale * marginal_message
            message = message.clamp(-self.message_clip, self.message_clip)
            shift = float(weight) * message
            real_logits = torch.where(
                is_masked[:, None], chunk[:, :self.category_sizes[index]] + shift,
                chunk[:, :self.category_sizes[index]],
            )
            guided_chunks.append(torch.cat(
                [real_logits, chunk[:, self.category_sizes[index]:]], dim=-1))
            new_posterior = F.softmax(real_logits, dim=-1)
            safe_code = current_codes[:, index].clamp(max=self.category_sizes[index] - 1)
            observed = F.one_hot(
                safe_code, num_classes=self.category_sizes[index]).to(new_posterior.dtype)
            new_belief = torch.where(is_masked[:, None], new_posterior, observed)
            guided_beliefs.append(new_belief)
            before_tv.append(0.5 * (belief.mean(dim=0) - target).abs().sum())
            after_tv.append(0.5 * (new_belief.mean(dim=0) - target).abs().sum())
            active_shift = shift[is_masked]
            if active_shift.numel():
                shifts.append(active_shift.abs().mean())
            event_count += int((is_masked & (shift.abs().amax(dim=1) > 1e-8)).sum())

        masked_cells = sum(int(value.sum()) for value in masked)
        possible_cells = current_codes.shape[0] * max(len(self.category_sizes), 1)
        pair_energy_before = 0.0
        pair_energy_after = 0.0
        pair_weight = 0.0
        for pair in self.pairs:
            pmi = pair.pmi.to(device=raw_logits.device, dtype=raw_logits.dtype)
            before = torch.einsum(
                "bi,ij,bj->b", beliefs[pair.left], pmi, beliefs[pair.right]).mean()
            after = torch.einsum(
                "bi,ij,bj->b", guided_beliefs[pair.left], pmi,
                guided_beliefs[pair.right]).mean()
            pair_energy_before += pair.normalized_mi * float(before)
            pair_energy_after += pair.normalized_mi * float(after)
            pair_weight += pair.normalized_mi
        if pair_weight > 0:
            pair_energy_before /= pair_weight
            pair_energy_after /= pair_weight
        diagnostics = {
            "categorical_guidance_events": event_count,
            "categorical_guidance_masked_fraction": masked_cells / possible_cells,
            "categorical_guidance_mean_logit_shift": float(
                torch.stack(shifts).mean()) if shifts else 0.0,
            "categorical_guidance_tv_before": float(torch.stack(before_tv).mean()),
            "categorical_guidance_tv_after": float(torch.stack(after_tv).mean()),
            "categorical_guidance_pairs": len(self.pairs),
            "categorical_guidance_pair_energy_before": pair_energy_before,
            "categorical_guidance_pair_energy_after": pair_energy_after,
        }
        return torch.cat(guided_chunks, dim=-1), diagnostics
