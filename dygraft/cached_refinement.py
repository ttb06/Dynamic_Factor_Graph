"""Drop-in refinement CLI that reuses train and anchor factor histograms."""
import math

import numpy as np

from dygraft import quality_constraints as evaluator
from dygraft import refinement as refinement
from dygraft.factor_statistics import smoothed_distribution


class FactorHistograms:
    def __init__(self, factors, anchor, pool_size):
        self.anchor = anchor
        self.anchor_mask = np.zeros(pool_size, dtype=bool)
        self.anchor_mask[anchor] = True
        self.work_mask = np.zeros(pool_size, dtype=bool)
        self.targets = [smoothed_distribution(f.train_states) for f in factors]
        self.anchor_counts = [np.bincount(f.pool_states[anchor], minlength=3) for f in factors]

    def score(self, indices, factors, support_tolerance, hard_tolerance):
        # The frozen draw generator builds full indices. Compute the membership
        # delta once, then apply it to the three-state counts of each factor.
        added = indices[~self.anchor_mask[indices]]
        self.work_mask[indices] = True
        removed = self.anchor[~self.work_mask[self.anchor]]
        self.work_mask[indices] = False
        relation_numerator = coverage_numerator = hard_numerator = 0.0
        total_weight = hard_weight = 0.0
        records = []
        for factor, target, anchor_counts in zip(factors, self.targets, self.anchor_counts):
            states = factor.pool_states
            counts = (anchor_counts - np.bincount(states[removed], minlength=3)
                      + np.bincount(states[added], minlength=3))
            smoothed = counts.astype(float) + 0.5
            observed = smoothed / smoothed.sum()
            relation = float(np.abs(target - observed).sum())
            active = counts[1] + counts[2]
            observed_support = float(active / len(indices))
            support_deficit = max(factor.train_support - support_tolerance - observed_support, 0.0)
            coverage = support_deficit / math.sqrt(max(factor.train_support, 1e-8))
            observed_violation = float(counts[2] / max(active, 1))
            hard_excess = (
                max(observed_violation - factor.train_violation_rate - hard_tolerance, 0.0)
                if factor.hard else 0.0)
            relation_numerator += factor.weight * relation
            coverage_numerator += factor.weight * coverage
            total_weight += factor.weight
            if factor.hard:
                hard_numerator += factor.weight * hard_excess
                hard_weight += factor.weight
            records.append({
                "candidate_id": factor.candidate_id,
                "weight": factor.weight,
                "hard": factor.hard,
                "relation_l1": relation,
                "train_support": factor.train_support,
                "selected_support": observed_support,
                "support_deficit": support_deficit,
                "train_violation_rate": factor.train_violation_rate,
                "selected_violation_rate": observed_violation,
                "hard_excess": hard_excess,
            })
        relation_mean = relation_numerator / max(total_weight, 1e-12)
        coverage_mean = coverage_numerator / max(total_weight, 1e-12)
        hard_mean = hard_numerator / max(hard_weight, 1e-12) if hard_weight else 0.0
        return float(relation_mean), float(coverage_mean), float(hard_mean), records


if __name__ == "__main__":
    original_repair = refinement.repair

    def cached_repair(anchor_indices, factors, generic_groups, args, geometry, integrated):
        scorer = FactorHistograms(factors, anchor_indices, args.pool_rows)
        original_loss = evaluator.graph_loss
        evaluator.graph_loss = scorer.score
        try:
            return original_repair(anchor_indices, factors, generic_groups, args, geometry, integrated)
        finally:
            evaluator.graph_loss = original_loss

    refinement.repair = cached_repair
    raise SystemExit(refinement.main())
