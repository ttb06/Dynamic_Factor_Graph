"""Exact greedy support repair with cached coverage and moment counts."""

import numpy as np

from dygraft import anchoring as bpr
class CoverageIndex:
    """Train neighborhoods incident to each candidate, using the original kNN radius."""

    def __init__(self, geometry, selected):
        feasible = geometry.nearest_distances <= geometry.real_radii[:, None]
        self.train_rows, columns = np.nonzero(feasible)
        self.pool_ids = geometry.nearest_indices[self.train_rows, columns]
        self.pool_size = len(selected)
        self.train_size = len(geometry.real_radii)
        order = np.argsort(self.pool_ids, kind="stable")
        self.members = self.train_rows[order]
        self.offsets = np.r_[0, np.bincount(self.pool_ids, minlength=self.pool_size).cumsum()]
        self.coverage = np.bincount(self.train_rows[selected[self.pool_ids]], minlength=self.train_size)

    def neighborhoods(self, candidate):
        return self.members[self.offsets[candidate]:self.offsets[candidate + 1]]

    def ranking(self, selected):
        uncovered = self.coverage[self.train_rows] == 0
        gains = np.bincount(self.pool_ids[uncovered], minlength=self.pool_size)
        gains[selected] = -1
        sole = (self.coverage[self.train_rows] == 1) & selected[self.pool_ids]
        owners = np.full(self.train_size, -1, dtype=np.int64)
        owners[self.train_rows[sole]] = self.pool_ids[sole]
        exclusive = np.bincount(self.pool_ids[sole], minlength=self.pool_size)
        return gains, exclusive, owners

    def rescued_by(self, candidate, owners):
        # A neighbor with exactly one selected member is rescued when the
        # proposed addition covers it and that selected member is removed.
        incumbent = owners[self.neighborhoods(candidate)]
        return np.bincount(incumbent[incumbent >= 0], minlength=self.pool_size)

    def accept(self, add, remove):
        self.coverage[self.neighborhoods(remove)] -= 1
        self.coverage[self.neighborhoods(add)] += 1


class MomentCounts:
    """Reuse each group's exact histogram and the original loss reductions."""

    def __init__(self, groups, anchor, pair_weight, coverage_weight):
        self.groups = groups
        self.targets = [bpr.target_distribution(group) for group in groups]
        self.scales = [np.sqrt(np.clip(target, 1e-8, None)) for target in self.targets]
        self.counts = [
            np.bincount(group.pool_codes[anchor], minlength=group.cardinality)
            for group in groups
        ]
        self.size = float(len(anchor))
        self.pair_weight = pair_weight
        self.coverage_weight = coverage_weight

    def score(self, add, remove):
        families = {"univariate": [], "pair": []}
        coverage = []
        for group, counts, target, scale in zip(self.groups, self.counts, self.targets, self.scales):
            observed = counts.astype(float)
            observed[group.pool_codes[remove]] -= 1
            observed[group.pool_codes[add]] += 1
            observed /= self.size
            residual = target - observed
            families[group.family].append(float(np.abs(residual).sum()))
            coverage.append(float(np.mean(np.maximum(residual, 0.0) / scale)))
        univariate = float(np.mean(families["univariate"])) if families["univariate"] else 0.0
        pair = float(np.mean(families["pair"])) if families["pair"] else 0.0
        support = float(np.mean(coverage)) if coverage else 0.0
        return univariate + self.pair_weight * pair + self.coverage_weight * support

    def accept(self, add, remove):
        for group, counts in zip(self.groups, self.counts):
            counts[group.pool_codes[remove]] -= 1
            counts[group.pool_codes[add]] += 1


def search(args, geometry, matrix, groups, selected, anchor, quality_limit, target_swaps, diagnostics):
    """Mutate the selected mask and return the original-format swap trace."""
    neighborhoods = CoverageIndex(geometry, selected)
    moments = MomentCounts(groups, anchor, args.pair_weight, args.coverage_weight)
    trace = []
    proposals_scanned = 0
    guard_checks = 0
    for step in range(target_swaps):
        covered = np.count_nonzero(neighborhoods.coverage)
        gains, exclusive, owners = neighborhoods.ranking(selected)
        positive = np.flatnonzero(gains > 0)
        if not len(positive):
            break
        additions = positive[np.argsort(-gains[positive], kind="stable")][:args.addition_shortlist]
        selected_indices = np.flatnonzero(selected)
        proposals = []
        for add in additions:
            distances = np.square(matrix[selected_indices] - matrix[add]).sum(axis=1)
            nearest = np.argsort(distances, kind="stable")[:args.removal_shortlist * 4]
            removals = sorted(
                selected_indices[nearest],
                key=lambda index: (
                    exclusive[index], float(np.square(matrix[index] - matrix[add]).sum()), int(index)
                ),
            )[:args.removal_shortlist]
            proposals_scanned += len(removals)
            rescued = neighborhoods.rescued_by(add, owners)
            for remove in removals:
                new_count = covered + gains[add] - exclusive[remove] + rescued[remove]
                if new_count > covered:
                    proposals.append((new_count, add, remove))

        # Coverage is the first element of the original key. Stable sorting
        # retains the original add/remove order for ties. Once a feasible
        # swap exists at a coverage level, lower levels cannot win.
        best = None
        for count, add, remove in sorted(proposals, key=lambda proposal: -proposal[0]):
            if best is not None and count < best[0][0]:
                break
            guard_checks += 1
            loss = moments.score(add, remove)
            if loss > quality_limit + 1e-12:
                continue
            key = (count, -loss, int(gains[add]), -exclusive[remove])
            if best is None or key > best[0]:
                best = (key, add, remove)
        if best is None or best[0][0] / neighborhoods.train_size <= covered / neighborhoods.train_size + 1e-12:
            break
        (count, neg_loss, gain, neg_exclusive), add, remove = best
        selected[remove] = False
        selected[add] = True
        neighborhoods.accept(add, remove)
        moments.accept(add, remove)
        trace.append({
            "step": step + 1, "add_index": int(add), "remove_index": int(remove),
            "uncovered_gain": gain, "exclusive_loss": int(-neg_exclusive),
            "geometry_coverage": float(count / neighborhoods.train_size), "objective": -neg_loss,
        })
    diagnostics.update({"proposals_scanned": proposals_scanned, "guard_checks": guard_checks})
    return trace
