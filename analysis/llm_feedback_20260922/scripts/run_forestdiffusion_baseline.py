#!/usr/bin/env python3
"""Run the official ForestDiffusion implementation on a TabDiff split.

The target is modelled jointly with the predictors.  This is important for a
generator comparison: passing it as ``label_y`` would turn the run into a
class-conditional experiment, while all other methods in the table learn the
unconditional joint table distribution.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--forest-repo", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("adult", "default", "shoppers", "magic", "diabetes", "beijing", "news"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--n-t", type=int, default=50)
    parser.add_argument("--duplicate-k", type=int, default=100)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--n-batch", type=int, default=10)
    parser.add_argument("--gpu-hist", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import importlib.util
    import sys
    import types
    package = args.forest_repo / "Python-Package" / "base-ForestDiffusion"
    sys.path.insert(0, str(package))
    # Upstream imports all optional estimator backends at module import time.
    # The experiment uses only XGBoost; lightweight placeholders avoid forcing
    # unrelated LightGBM/CatBoost installations without changing the executed
    # ForestDiffusion code path.
    if importlib.util.find_spec("lightgbm") is None:
        module = types.ModuleType("lightgbm")
        module.LGBMRegressor = type("UnavailableLGBMRegressor", (), {})
        sys.modules["lightgbm"] = module
    if importlib.util.find_spec("catboost") is None:
        module = types.ModuleType("catboost")
        module.CatBoostRegressor = type("UnavailableCatBoostRegressor", (), {})
        sys.modules["catboost"] = module
    from ForestDiffusion import ForestDiffusionModel

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train = pd.read_csv(args.repo / "data" / args.dataset / "train.csv")
    columns = list(train.columns)

    categorical_positions = list(info["cat_col_idx"]) + list(info["target_col_idx"])
    integer_positions = list(info.get("int_col_idx", []))
    category_values: dict[int, list[str]] = {}
    encoded = train.copy()
    for position in categorical_positions:
        values = train.iloc[:, position].astype(str)
        categories = sorted(values.unique().tolist())
        category_values[position] = categories
        lookup = {value: index for index, value in enumerate(categories)}
        encoded.iloc[:, position] = values.map(lookup).astype(float)

    x = encoded.to_numpy(dtype=float)
    binary = [position for position in categorical_positions if len(category_values[position]) == 2]
    categorical = [position for position in categorical_positions if len(category_values[position]) > 2]
    integer = [position for position in integer_positions if position not in binary]

    start = time.time()
    model = ForestDiffusionModel(
        x,
        n_t=args.n_t,
        duplicate_K=args.duplicate_k,
        bin_indexes=binary,
        cat_indexes=categorical,
        int_indexes=integer,
        diffusion_type="flow",
        model="xgboost",
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        n_batch=args.n_batch,
        gpu_hist=args.gpu_hist,
        seed=args.seed,
    )
    generated = model.generate(batch_size=args.samples, n_t=args.n_t)
    elapsed = time.time() - start

    result = pd.DataFrame(generated, columns=columns)
    for position, categories in category_values.items():
        ids = np.rint(result.iloc[:, position].to_numpy(float)).astype(int)
        ids = np.clip(ids, 0, len(categories) - 1)
        result.iloc[:, position] = np.asarray(categories, dtype=object)[ids]
        # Restore numeric categorical dtypes when the original table used them.
        original = train.iloc[:, position]
        if pd.api.types.is_bool_dtype(original):
            result.iloc[:, position] = (
                result.iloc[:, position].astype(str).map({"True": True, "False": False}).astype(bool)
            )
        elif pd.api.types.is_numeric_dtype(original):
            result.iloc[:, position] = pd.to_numeric(result.iloc[:, position])
    for position in integer_positions:
        result.iloc[:, position] = np.rint(pd.to_numeric(result.iloc[:, position])).astype(float)

    sample_path = args.output / "forest_diffusion.csv"
    result.to_csv(sample_path, index=False)
    metadata = {
        "implementation": "official SamsungSAILMontreal/ForestDiffusion",
        "dataset": args.dataset,
        "seed": args.seed,
        "samples": args.samples,
        "n_t": args.n_t,
        "duplicate_k": args.duplicate_k,
        "n_estimators": args.n_estimators,
        "n_batch": args.n_batch,
        "gpu_hist": args.gpu_hist,
        "joint_target_modelling": True,
        "runtime_seconds": elapsed,
        "binary_positions": binary,
        "categorical_positions": categorical,
        "integer_positions": integer,
        "sample_path": str(sample_path),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
