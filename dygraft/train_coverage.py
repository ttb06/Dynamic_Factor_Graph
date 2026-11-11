"""Reuse the fixed train reference across checkpoints in a coverage fold."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from dygraft.anchoring import typed_columns


class TrainCoverageFold:
    def __init__(self, train, info, synthetic_size, seed):
        self.numeric, self.categorical = typed_columns(info)
        self.synthetic_size = synthetic_size
        n = min(len(train), synthetic_size)
        rng = np.random.default_rng(seed)
        real = train.iloc[rng.choice(len(train), n, replace=False)].reset_index(drop=True)
        self.synthetic_order = rng.choice(synthetic_size, n, replace=False)
        self.encoder = None
        if self.categorical:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            real_values = real[self.categorical].astype(str)
            self.encoder.fit(real_values)
            real_cat = self.encoder.transform(real_values)
        else:
            real_cat = np.empty((n, 0), dtype=float)
        real_num = real[self.numeric].to_numpy(float)
        real_matrix = np.concatenate([real_num, real_cat], axis=1)
        self.low = np.nanmin(real_matrix, axis=0)
        self.span = np.nanmax(real_matrix, axis=0) - self.low
        self.span[self.span == 0] = 1.0
        self.real_matrix = (real_matrix - self.low) / self.span
        self.center = np.mean(self.real_matrix, axis=0)
        self.alphas = np.linspace(0, 1, 30)
        self.radii = np.quantile(np.linalg.norm(self.real_matrix - self.center, axis=1), self.alphas)
        real_nn = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(self.real_matrix)
        self.real_to_real = real_nn.kneighbors(self.real_matrix, return_distance=True)[0][:, 1]

    def score(self, synthetic):
        if len(synthetic) != self.synthetic_size:
            raise ValueError("coverage fold requires the same synthetic cardinality at every checkpoint")
        synthetic = synthetic.iloc[self.synthetic_order].reset_index(drop=True)
        if self.encoder is not None:
            synthetic_cat = self.encoder.transform(synthetic[self.categorical].astype(str))
        else:
            synthetic_cat = np.empty((len(synthetic), 0), dtype=float)
        synthetic_num = synthetic[self.numeric].to_numpy(float)
        synthetic_matrix = np.concatenate([synthetic_num, synthetic_cat], axis=1)
        synthetic_matrix = (synthetic_matrix - self.low) / self.span
        synthetic_center = np.mean(synthetic_matrix, axis=0)
        synthetic_to_center = np.linalg.norm(synthetic_matrix - self.center, axis=1)
        synthetic_nn = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(synthetic_matrix)
        real_to_synthetic, nearest = synthetic_nn.kneighbors(self.real_matrix, return_distance=True)
        real_to_synthetic = real_to_synthetic[:, 0]
        closest = synthetic_matrix[nearest[:, 0]]
        closest_radius = np.linalg.norm(closest - synthetic_center, axis=1)
        closest_quantiles = np.quantile(closest_radius, self.alphas)
        precision_curve = np.asarray([np.mean(synthetic_to_center <= radius) for radius in self.radii])
        coverage_curve = np.asarray([
            np.mean((real_to_synthetic <= self.real_to_real) & (closest_radius <= radius))
            for radius in closest_quantiles])
        alpha_precision = 1.0 - np.abs(self.alphas - precision_curve).sum() / self.alphas.sum()
        beta_recall = 1.0 - np.abs(self.alphas - coverage_curve).sum() / self.alphas.sum()
        return float(alpha_precision), float(beta_recall)
