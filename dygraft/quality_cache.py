"""Cache bin contributions while preserving the original loss reductions."""
import numpy as np

from dygraft import anchoring as bpr
class CachedGuardTerms:
    def __init__(self, groups, anchor, pair_weight, coverage_weight):
        self.size = float(len(anchor))
        self.pair_weight = pair_weight
        self.coverage_weight = coverage_weight
        self.univariate = np.array([g.family == "univariate" for g in groups], dtype=bool)
        self.slices = []
        if not groups:
            return
        sizes = np.array([g.cardinality for g in groups], dtype=np.intp)
        starts = np.r_[0, sizes.cumsum()]
        self.slices = [slice(start, end) for start, end in zip(starts, starts[1:])]
        self.bin_codes = np.stack([g.pool_codes for g in groups], axis=1) + starts[:-1]
        self.counts = np.bincount(self.bin_codes[anchor].ravel(), minlength=starts[-1])
        self.targets = np.concatenate([bpr.target_distribution(group) for group in groups])
        self.scales = np.sqrt(np.clip(self.targets, 1e-8, None))
        self._refresh_terms()

    def _refresh_terms(self):
        observed = self.counts.astype(float) + np.array([[-1.0], [0.0], [1.0]])
        observed /= self.size
        residual = self.targets - observed
        absolute = np.abs(residual)
        coverage = np.maximum(residual, 0.0) / self.scales
        # Each column stores the absolute and coverage term for a single bin.
        self.minus = np.stack((absolute[0], coverage[0]))
        self.current = np.stack((absolute[1], coverage[1]))
        self.plus = np.stack((absolute[2], coverage[2]))
        self.work = self.current.copy()

    def _changed_bins(self, add, remove):
        before = self.bin_codes[remove]
        after = self.bin_codes[add]
        changed = before != after
        return before[changed], after[changed]

    def score(self, add, remove):
        if not self.slices:
            return 0.0
        before, after = self._changed_bins(add, remove)
        self.work[:, before] = self.minus[:, before]
        self.work[:, after] = self.plus[:, after]
        absolute, coverage = self.work
        try:
            # Use the original contiguous per-group reductions and group order.
            # Delta sums of pre-aggregated scores could change close decisions.
            l1 = np.array([absolute[bins].sum() for bins in self.slices])
            gaps = np.array([coverage[bins].mean() for bins in self.slices])
            univariate = l1[self.univariate]
            pair = l1[~self.univariate]
            univariate_mean = float(univariate.mean()) if len(univariate) else 0.0
            pair_mean = float(pair.mean()) if len(pair) else 0.0
            return univariate_mean + self.pair_weight * pair_mean + self.coverage_weight * float(gaps.mean())
        finally:
            self.work[:, before] = self.current[:, before]
            self.work[:, after] = self.current[:, after]

    def accept(self, add, remove):
        if not self.slices:
            return
        before, after = self._changed_bins(add, remove)
        self.counts[before] -= 1
        self.counts[after] += 1
        self._refresh_terms()
