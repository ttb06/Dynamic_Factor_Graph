"""Runtime helpers extracted from the experiment implementation."""
from __future__ import annotations

import json


from pathlib import Path


import numpy as np


import torch


from sklearn.pipeline import Pipeline


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler


class AnalysisDataset:
    """Minimal reproduction of TabDiff's Adult preprocessing.

    Keeping this local avoids importing evaluation-only dependencies from the
    upstream ``src`` package in the GPU environment.
    """

    def __init__(self, x: torch.Tensor, num_transform, cat_transform, categories: np.ndarray,
                 d_numerical: int, int_inverse=None):
        self.X = x
        self.num_transform = num_transform
        self.cat_transform = cat_transform
        self._int_inverse = int_inverse
        self.categories = categories
        self.d_numerical = d_numerical

    def __len__(self):
        return len(self.X)

    def num_inverse(self, values):
        if hasattr(self.num_transform, "inverse_transform"):
            return self.num_transform.inverse_transform(values)
        return self.num_transform(values)

    def int_inverse(self, values):
        return self._int_inverse(values) if self._int_inverse is not None else values

    def cat_inverse(self, values):
        if not hasattr(self.cat_transform, "named_steps"):
            return self.cat_transform(values)
        array = np.asarray(values).copy()
        encoder = self.cat_transform.named_steps["ordinal"]
        invalid = np.zeros(array.shape, dtype=bool)
        safe = array.copy()
        for column, categories in enumerate(encoder.categories_):
            column_invalid = (~np.isfinite(safe[:, column])) | (safe[:, column] < 0) | (safe[:, column] >= len(categories))
            invalid[:, column] = column_invalid
            safe[column_invalid, column] = 0
        decoded = self.cat_transform.inverse_transform(safe).astype(object)
        decoded[invalid] = "<UNKNOWN>"
        return decoded


def preprocess_dataset(repo: Path, dataset: str,
                       regression_target_mode: str = "categorical",
                       ) -> tuple[AnalysisDataset, AnalysisDataset]:
    """Reproduce the local TabDiff preprocessing for any prepared dataset.

    The prepared Adult, Default, and Shoppers directories share the same
    X_num/X_cat/y layout.  Keeping one train-fitted transform here also avoids
    accidental held-out leakage when adding cross-dataset pilots.
    """
    data = repo / "data" / dataset
    info = json.loads((data / "info.json").read_text())
    if regression_target_mode not in {"categorical", "numerical"}:
        raise ValueError("regression_target_mode must be categorical or numerical")
    x_num_train = np.load(data / "X_num_train.npy", allow_pickle=True).astype(float)
    x_num_test = np.load(data / "X_num_test.npy", allow_pickle=True).astype(float)
    x_cat_train = np.load(data / "X_cat_train.npy", allow_pickle=True).astype(str)
    x_cat_test = np.load(data / "X_cat_test.npy", allow_pickle=True).astype(str)
    if info["task_type"] == "regression" and regression_target_mode == "numerical":
        # Exact released convention: regression y is the first numerical
        # feature.  The transform parameters below mirror src.Transformations.
        y_train = np.load(data / "y_train.npy", allow_pickle=True).reshape(-1, 1).astype(float)
        y_test = np.load(data / "y_test.npy", allow_pickle=True).reshape(-1, 1).astype(float)
        x_num_train = np.concatenate([y_train, x_num_train], axis=1)
        x_num_test = np.concatenate([y_test, x_num_test], axis=1)
    else:
        # Backward-compatible mode for already trained mechanism checkpoints.
        y_train = np.load(data / "y_train.npy", allow_pickle=True).reshape(-1, 1).astype(str)
        y_test = np.load(data / "y_test.npy", allow_pickle=True).reshape(-1, 1).astype(str)
        x_cat_train = np.concatenate([y_train, x_cat_train], axis=1)
        x_cat_test = np.concatenate([y_test, x_cat_test], axis=1)

    means = np.nanmean(x_num_train, axis=0)
    for array in (x_num_train, x_num_test):
        where = np.where(np.isnan(array))
        array[where] = np.take(means, where[1])
    num_transform = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=max(min(len(x_num_train) // 30, 1000), 10),
        subsample=int(1e9),
        random_state=0,
    ).fit(x_num_train)
    train_num = num_transform.transform(x_num_train)
    test_num = num_transform.transform(x_num_test)

    unknown = np.iinfo("int64").max - 3
    cat_transform = Pipeline(
        [("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=unknown, dtype="int64"))]
    ).fit(x_cat_train)
    train_cat = cat_transform.transform(x_cat_train)
    test_cat = cat_transform.transform(x_cat_test)
    max_values = train_cat.max(axis=0)
    for column in range(test_cat.shape[1]):
        test_cat[test_cat[:, column] == unknown, column] = max_values[column] + 1
    categories = np.array([len(set(train_cat[:, i].tolist())) for i in range(train_cat.shape[1])])
    train_tensor = torch.tensor(np.concatenate([train_num, train_cat], axis=1)).float()
    test_tensor = torch.tensor(np.concatenate([test_num, test_cat], axis=1)).float()
    return (
        AnalysisDataset(train_tensor, num_transform, cat_transform, categories, train_num.shape[1]),
        AnalysisDataset(test_tensor, num_transform, cat_transform, categories, train_num.shape[1]),
    )


def preprocess_adult(repo: Path) -> tuple[AnalysisDataset, AnalysisDataset]:
    """Backward-compatible alias used by the original Adult probes."""
    return preprocess_dataset(repo, "adult")
