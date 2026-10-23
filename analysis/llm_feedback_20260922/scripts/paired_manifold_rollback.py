#!/usr/bin/env python3
"""Select between paired certified samples using a train-only manifold score.

The two inputs must be row-aligned samples generated from the same initial
noise.  The selector never sees held-out rows or downstream evaluation
metrics.  It embeds mixed-type train rows, fits train prototypes, and keeps
the trajectory-guided candidate only when it is at least as well supported as
the final-repair reference.  This is a conservative rollback controller, not
a new generator or a post-hoc test-set optimiser.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler


def numeric_and_categorical(train: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in train:
        converted = pd.to_numeric(train[column], errors="coerce")
        if converted.notna().mean() >= 0.98:
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def aligned(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    stripped = {str(column).strip(): column for column in frame.columns}
    wanted = {str(column).strip(): column for column in columns}
    missing = [name for name in wanted if name not in stripped]
    if missing:
        raise ValueError(f"sample is missing columns: {missing}")
    result = frame[[stripped[name] for name in wanted]].copy()
    result.columns = columns
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Paired final-repair sample.")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Paired trajectory-guided and certified sample.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototypes", type=int, default=256)
    parser.add_argument("--components", type=int, default=20)
    parser.add_argument("--relative-slack", type=float, default=0.0,
                        help="Allow candidate distance <= reference*(1+slack).")
    parser.add_argument("--seed", type=int, default=20261607)
    args = parser.parse_args()
    if args.relative_slack < 0:
        raise ValueError("relative slack must be non-negative")

    train = pd.read_csv(args.train)
    reference = aligned(pd.read_csv(args.reference), list(train.columns))
    candidate = aligned(pd.read_csv(args.candidate), list(train.columns))
    if len(reference) != len(candidate):
        raise ValueError("paired inputs have different row counts")

    numeric, categorical = numeric_and_categorical(train)
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            make_pipeline(
                QuantileTransformer(
                    n_quantiles=min(512, len(train)), output_distribution="normal",
                    subsample=min(100_000, len(train)), random_state=args.seed,
                ),
                StandardScaler(),
            ),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))
    transform = ColumnTransformer(transformers, sparse_threshold=0.0)
    train_x = transform.fit_transform(train)
    reference_x = transform.transform(reference)
    candidate_x = transform.transform(candidate)

    components = min(args.components, train_x.shape[1], len(train_x) - 1)
    if components > 0 and train_x.shape[1] > components:
        pca = PCA(n_components=components, whiten=True, random_state=args.seed)
        train_x = pca.fit_transform(train_x)
        reference_x = pca.transform(reference_x)
        candidate_x = pca.transform(candidate_x)

    prototypes = min(max(args.prototypes, 2), len(train))
    model = MiniBatchKMeans(
        n_clusters=prototypes, batch_size=min(4096, len(train)), n_init=3,
        max_iter=100, random_state=args.seed,
    ).fit(train_x)
    train_labels = model.labels_
    train_distance = np.linalg.norm(train_x - model.cluster_centers_[train_labels], axis=1)
    global_radius = max(float(np.median(train_distance)), 1e-8)
    radii = np.full(prototypes, global_radius, dtype=float)
    for label in range(prototypes):
        local = train_distance[train_labels == label]
        if len(local) >= 5:
            radii[label] = max(float(np.median(local)), 1e-8)

    def support_distance(values: np.ndarray) -> np.ndarray:
        raw = model.transform(values)
        labels = raw.argmin(axis=1)
        return raw[np.arange(len(values)), labels] / radii[labels]

    reference_distance = support_distance(reference_x)
    candidate_distance = support_distance(candidate_x)
    changed = (reference.astype(str).to_numpy() != candidate.astype(str).to_numpy()).any(axis=1)
    select = changed & (
        candidate_distance <= reference_distance * (1.0 + args.relative_slack)
    )

    result = reference.copy()
    result.loc[select, :] = candidate.loc[select, :].to_numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    diagnostics = {
        "rows": len(result),
        "changed_candidate_rows": int(changed.sum()),
        "selected_candidate_rows": int(select.sum()),
        "selected_candidate_fraction": float(select.mean()),
        "selected_among_changed_fraction": float(select.sum() / max(changed.sum(), 1)),
        "mean_reference_support_distance": float(reference_distance.mean()),
        "mean_candidate_support_distance": float(candidate_distance.mean()),
        "mean_selected_support_improvement": float(
            (reference_distance[select] - candidate_distance[select]).mean()
            if select.any() else 0.0
        ),
        "prototypes": prototypes,
        "components": components,
        "relative_slack": args.relative_slack,
        "heldout_used": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
