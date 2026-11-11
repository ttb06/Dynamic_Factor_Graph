"""Screen support swaps in batches, then score possible winners exactly."""
from itertools import groupby
import os

import numpy as np

from dygraft import support as repair
from dygraft.coverage_index import CoverageIndex

from dygraft.quality_cache import CachedGuardTerms


class ScreenedGuard(CachedGuardTerms):
    def __init__(self, groups, anchor, pair_weight, coverage_weight):
        super().__init__(groups, anchor, pair_weight, coverage_weight)
        bins = np.array([group.cardinality for group in groups], dtype=np.intp)
        uni = np.count_nonzero(self.univariate)
        pair = len(groups) - uni
        coefficients = np.where(
            self.univariate, 1.0 / max(uni, 1), pair_weight / max(pair, 1),
        )
        self.weights = np.stack((np.repeat(coefficients, bins),
                                 np.repeat(coverage_weight / (max(len(groups), 1) * bins), bins)))
        # Cushion for re-association of normalized, bounded per-bin terms.
        # Near-best proposals and those near the quality threshold are
        # rescored with the original contiguous reductions.
        self.margin = (256 * np.finfo(float).eps * (bins.sum() + len(groups) + 1)
                       * (1 + abs(pair_weight) + abs(coverage_weight)))
        self._refresh_screen()

    def _refresh_screen(self):
        self.base = super().score(0, 0)
        if self.slices:
            self.subtract = ((self.minus - self.current) * self.weights).sum(axis=0)
            self.insert = ((self.plus - self.current) * self.weights).sum(axis=0)

    def estimates(self, additions, removals):
        if not self.slices:
            return np.zeros(len(additions))
        before, after = self.bin_codes[removals], self.bin_codes[additions]
        delta = np.where(before != after, self.subtract[before] + self.insert[after], 0.0)
        return self.base + delta.sum(axis=1)

    def accept(self, add, remove):
        super().accept(add, remove)
        self._refresh_screen()


def closest_indices(distances, limit):
    """Use selection, breaking exact distance ties by ascending input index."""
    if not np.isfinite(distances).all():
        return np.argsort(distances, kind="stable")[:limit]
    if limit >= len(distances):
        return np.arange(len(distances))
    cutoff = np.partition(distances, limit - 1)[limit - 1]
    smaller = np.flatnonzero(distances < cutoff)
    tied = np.flatnonzero(distances == cutoff)
    return np.r_[smaller, tied[:limit-len(smaller)]]


def search(args, geometry, matrix, groups, selected, anchor, quality_limit, target_swaps, diagnostics):
    neighborhoods = CoverageIndex(geometry, selected)
    screened = os.environ["DYGRAFT_SCREEN_SUPPORT"] == "1"
    guard_cls = ScreenedGuard if screened else CachedGuardTerms
    guard = guard_cls(groups, anchor, args.pair_weight, args.coverage_weight)
    trace = []
    proposals_scanned = guard_checks = exact_guard_checks = 0
    ceiling = quality_limit + 1e-12
    for step in range(target_swaps):
        covered = np.count_nonzero(neighborhoods.coverage)
        gains, exclusive, owners = neighborhoods.ranking(selected)
        positive = np.flatnonzero(gains > 0)
        if not len(positive):
            break
        additions = positive[np.argsort(-gains[positive], kind="stable")][:args.addition_shortlist]
        selected_indices = np.flatnonzero(selected)
        selected_matrix = matrix[selected_indices]
        proposals = []
        for add in additions:
            distances = np.square(selected_matrix - matrix[add]).sum(axis=1)
            nearest = closest_indices(distances, args.removal_shortlist * 4)
            removals = sorted(selected_indices[nearest], key=lambda index: (
                exclusive[index], float(np.square(matrix[index] - matrix[add]).sum()), int(index),
            ))[:args.removal_shortlist]
            proposals_scanned += len(removals)
            rescued = neighborhoods.rescued_by(add, owners)
            for remove in removals:
                count = covered + gains[add] - exclusive[remove] + rescued[remove]
                if count > covered:
                    proposals.append((count, add, remove))

        best = None
        ordered = sorted(proposals, key=lambda proposal: -proposal[0])
        for count, group in groupby(ordered, key=lambda proposal: proposal[0]):
            pairs = [(add, remove) for _, add, remove in group]
            guard_checks += len(pairs)
            candidates = range(len(pairs))
            if screened:
                add_ids, remove_ids = np.asarray(pairs, dtype=np.intp).T
                scores = guard.estimates(add_ids, remove_ids)
                if np.isfinite(scores).all():
                    candidates = np.flatnonzero(
                        (scores <= ceiling + guard.margin)
                        & (scores <= scores.min() + 2 * guard.margin)
                    )
            for index in candidates:
                add, remove = pairs[index]
                exact_guard_checks += 1
                loss = guard.score(add, remove)
                if loss > ceiling:
                    continue
                key = (count, -loss, int(gains[add]), -exclusive[remove])
                if best is None or key > best[0]:
                    best = (key, add, remove)
            if best is not None:
                break
        if best is None or best[0][0] / neighborhoods.train_size <= covered / neighborhoods.train_size + 1e-12:
            break
        (count, neg_loss, gain, neg_exclusive), add, remove = best
        selected[remove] = False
        selected[add] = True
        neighborhoods.accept(add, remove)
        guard.accept(add, remove)
        trace.append({
            "step": step + 1, "add_index": int(add), "remove_index": int(remove),
            "uncovered_gain": gain, "exclusive_loss": int(-neg_exclusive),
            "geometry_coverage": float(count / neighborhoods.train_size), "objective": -neg_loss,
        })
    diagnostics.update(proposals_scanned=proposals_scanned, guard_checks=guard_checks,
                       exact_guard_checks=exact_guard_checks)
    return trace


if __name__ == "__main__":
    repair.search = search
    raise SystemExit(repair.main())
