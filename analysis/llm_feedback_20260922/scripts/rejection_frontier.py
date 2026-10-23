#!/usr/bin/env python3
"""Matched rejection, conditional generation, and certification frontier.

The reference cohort is decoded from the held-out split.  Contracts remain
controlled cohort requests, not asserted real-world invariants.  The script
loads a previously trained structured adapter and never retrains it.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

from conditioned_tabdiff_adapter import ConditionedUniModMLP, ContractConditionedModel, decode, seed_all, total_variation
from tabdiff_probe import preprocess_dataset


def contracts_for(dataset: str, catalog: str, train: pd.DataFrame | None = None) -> list[dict]:
    if catalog == "support_numeric":
        if train is None:
            raise ValueError("training frame is required for support sweep")
        from cross_dataset_condition_adapter import CATALOGS

        result = []
        for column in CATALOGS[dataset]["scales"]:
            values = pd.to_numeric(train[column], errors="coerce")
            for quantile in (0.50, 0.75, 0.90, 0.95, 0.975, 0.99):
                threshold = float(values.quantile(quantile))
                result.append({
                    "contract_id": f"support_{column}_q{int(1000 * quantile):03d}",
                    "split": f"support_q{int(1000 * quantile):03d}",
                    "family": column,
                    "predicates": [(column, "ge", threshold)],
                })
        return result
    if catalog == "support_mixed":
        if train is None:
            raise ValueError("training frame is required for support sweep")
        from mixed_type_condition_adapter import SPECS

        numeric_columns = list(SPECS[dataset]["numeric"])
        categorical_columns = list(SPECS[dataset]["categorical"])
        result = []
        for numeric_column, categorical_column in zip(numeric_columns, categorical_columns):
            allowed = [str(value) for value in SPECS[dataset]["categorical"][categorical_column]]
            frequencies = train[categorical_column].astype(str).value_counts(normalize=True)
            candidates = [(float(frequencies.get(value, 0.0)), value) for value in allowed if float(frequencies.get(value, 0.0)) >= 0.01]
            if not candidates:
                raise RuntimeError(f"no supported category for {categorical_column}")
            category = min(candidates)[1]
            values = pd.to_numeric(train[numeric_column], errors="coerce")
            for quantile in (0.75, 0.90, 0.95, 0.975):
                threshold = float(values.quantile(quantile))
                result.append({
                    "contract_id": f"support_{numeric_column}_{categorical_column}_q{int(1000 * quantile):03d}",
                    "split": f"support_mixed_q{int(1000 * quantile):03d}",
                    "family": f"{numeric_column}_{categorical_column}",
                    "predicates": [(numeric_column, "ge", threshold), (categorical_column, "eq", category)],
                })
        return result
    if catalog == "numeric":
        from cross_dataset_condition_adapter import make_catalog

        raw = make_catalog(dataset)
        result = []
        for contract in raw:
            normalized = copy.deepcopy(contract)
            normalized["predicates"] = [(column, "ge", value) for column, value in contract["predicates"]]
            result.append(normalized)
        return [contract for contract in result if contract["split"] != "train"]
    from mixed_type_condition_adapter import make_catalog

    return [contract for contract in make_catalog(dataset) if contract["split"] != "train"]


def contract_vector(contract: dict, dataset: str, catalog: str) -> list[float]:
    if catalog in {"numeric", "support_numeric"}:
        from cross_dataset_condition_adapter import vector

        pair_contract = copy.deepcopy(contract)
        pair_contract["predicates"] = [(column, value) for column, operator, value in contract["predicates"] if operator == "ge"]
        return vector(pair_contract, dataset)
    from mixed_type_condition_adapter import structured_vector

    return structured_vector(contract, dataset)


def satisfies(frame: pd.DataFrame, contract: dict) -> pd.Series:
    selected = pd.Series(True, index=frame.index)
    for column, operator, value in contract["predicates"]:
        if operator == "ge":
            selected &= pd.to_numeric(frame[column], errors="coerce").ge(float(value))
        elif operator == "eq":
            selected &= frame[column].astype(str).str.strip().eq(str(value).strip())
        else:
            raise ValueError(operator)
    return selected


def fidelity(sample: pd.DataFrame, reference: pd.DataFrame, numeric: list[str], targets: set[str]) -> dict:
    if len(sample) == 0 or len(reference) == 0:
        return {
            "numeric_ks": math.nan,
            "categorical_tv": math.nan,
            "target_numeric_ks": math.nan,
            "nontarget_numeric_ks": math.nan,
            "target_categorical_tv": math.nan,
            "nontarget_categorical_tv": math.nan,
        }
    categorical = [column for column in reference.columns if column not in numeric]
    ks = {column: float(ks_2samp(pd.to_numeric(sample[column]), pd.to_numeric(reference[column])).statistic) for column in numeric}
    tv = {column: float(total_variation(sample[column], reference[column])) for column in categorical}

    def mean(values):
        return float(np.mean(values)) if values else math.nan

    return {
        "numeric_ks": mean(list(ks.values())),
        "categorical_tv": mean(list(tv.values())),
        "target_numeric_ks": mean([value for column, value in ks.items() if column in targets]),
        "nontarget_numeric_ks": mean([value for column, value in ks.items() if column not in targets]),
        "target_categorical_tv": mean([value for column, value in tv.items() if column in targets]),
        "nontarget_categorical_tv": mean([value for column, value in tv.items() if column not in targets]),
    }


def certify(sample: pd.DataFrame, contract: dict, heldout: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    result = sample.copy()
    edited = np.zeros(len(result), dtype=bool)
    distortions = []
    for column, operator, value in contract["predicates"]:
        if operator == "ge":
            old = pd.to_numeric(result[column]).copy()
            bad = old.lt(float(value))
            result.loc[bad, column] = float(value)
            scale = max(float(pd.to_numeric(heldout[column]).std()), 1e-6)
            distortions.append(float((pd.to_numeric(result[column]) - old).abs().mean() / scale))
        elif operator == "eq":
            bad = ~result[column].astype(str).str.strip().eq(str(value).strip())
            result.loc[bad, column] = str(value)
        else:
            raise ValueError(operator)
        edited |= bad.to_numpy()
    return result, float(edited.mean()), float(np.mean(distortions)) if distortions else 0.0


def sample_chunks(diffusion, count: int, chunk: int, seed: int, train_data, info) -> pd.DataFrame:
    frames = []
    for offset, start in enumerate(range(0, count, chunk)):
        size = min(chunk, count - start)
        seed_all(seed + offset)
        with torch.no_grad():
            frames.append(decode(train_data, diffusion.sample(size), info))
    return pd.concat(frames, ignore_index=True)


def add_result(rows: list[dict], sample: pd.DataFrame, reference: pd.DataFrame, numeric: list[str], contract: dict,
               method: str, dataset: str, catalog: str, seed: int, trajectories: int, edit_rate: float = 0.0,
               numeric_distortion: float = 0.0) -> None:
    targets = {column for column, _, _ in contract["predicates"]}
    row = {
        "dataset": dataset,
        "catalog": catalog,
        "seed": seed,
        "contract_id": contract["contract_id"],
        "split": contract["split"],
        "method": method,
        "rows": len(sample),
        "trajectories": trajectories,
        "satisfaction": float(satisfies(sample, contract).mean()) if len(sample) else math.nan,
        "edit_rate": edit_rate,
        "numeric_distortion": numeric_distortion,
        "reference_rows": len(reference),
    }
    row.update(fidelity(sample, reference, numeric, targets))
    rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", choices=["adult", "default", "shoppers"], required=True)
    parser.add_argument("--catalog", choices=["numeric", "mixed", "support_numeric", "support_mixed"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-samples", type=int, default=65536)
    parser.add_argument("--conditional-samples", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scales", default="1:2:4")
    parser.add_argument("--run-label", default="base")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    started = time.perf_counter()
    seed_all(args.seed)
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    device = torch.device("cuda")
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, args.dataset)
    train = decode(train_data, train_data.X, info)
    heldout = decode(train_data, test_data.X, info)
    numeric = [info["column_names"][index] for index in info["num_col_idx"]]
    contracts = contracts_for(args.dataset, args.catalog, train)
    saved = torch.load(args.adapter, map_location="cpu", weights_only=False)
    vectors = {contract["contract_id"]: contract_vector(contract, args.dataset, args.catalog) for contract in contracts}
    condition_dim = int(saved["condition_dim"])
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    backbone = ConditionedUniModMLP(condition_dim=condition_dim, **params)
    base = Model(backbone, **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = base.load_state_dict(checkpoint["denoise_fn"], strict=False)
    unexpected_missing = [key for key in incompatible.missing_keys if "condition_projection" not in key]
    if incompatible.unexpected_keys or unexpected_missing:
        raise RuntimeError(str(incompatible))
    backbone.mlp.condition_projection.load_state_dict(saved["condition_projection"])
    backbone.mlp.mlp.load_state_dict(saved["latent_mlp"])
    model = ContractConditionedModel(base, backbone, condition_dim).to(device)
    diffusion_parameters = copy.deepcopy(config["diffusion_params"])
    diffusion_parameters.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_parameters["sampler_params"] = copy.deepcopy(diffusion_parameters["sampler_params"])
    diffusion_parameters["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, model, None, **diffusion_parameters, device=device).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    diffusion.eval()

    zero = torch.zeros(args.chunk, condition_dim, device=device)
    model.set_condition(zero, 1.0)
    pool = sample_chunks(diffusion, args.pool_samples, args.chunk, args.seed + 100_000, train_data, info)
    result_rows = []
    budget_rows = []
    budgets = [1, 2, 4, 8, 16, 32, 64]
    guidance_values = [float(value) for value in args.guidance_scales.replace(",", ":").split(":") if value]

    for ordinal, contract in enumerate(contracts):
        reference = heldout[satisfies(heldout, contract)].copy()
        accepted_mask = satisfies(pool, contract)
        accepted = pool[accepted_mask].copy()
        acceptance = float(accepted_mask.mean())
        add_result(result_rows, pool, reference, numeric, contract, "unconditional", args.dataset, args.catalog, args.seed, args.pool_samples)
        add_result(result_rows, accepted, reference, numeric, contract, "rejection", args.dataset, args.catalog, args.seed, args.pool_samples)
        for budget in budgets:
            budget_rows.append({
                "dataset": args.dataset,
                "catalog": args.catalog,
                "seed": args.seed,
                "contract_id": contract["contract_id"],
                "split": contract["split"],
                "method": "rejection",
                "budget": budget,
                "single_trajectory_success": acceptance,
                "success_with_budget": 1.0 - (1.0 - acceptance) ** budget,
                "expected_trajectories_per_valid": math.inf if acceptance == 0 else 1.0 / acceptance,
                "heldout_support": float(len(reference) / len(heldout)),
            })

        for guidance in guidance_values:
            condition_batch = min(args.chunk, args.conditional_samples)
            condition = torch.tensor(vectors[contract["contract_id"]], dtype=torch.float32, device=device).repeat(condition_batch, 1)
            model.set_condition(condition, guidance)
            generated = sample_chunks(diffusion, args.conditional_samples, args.chunk, args.seed + 200_000 + ordinal * 100 + int(guidance * 10), train_data, info)
            valid = satisfies(generated, contract)
            method = f"condition_g{guidance:g}"
            add_result(result_rows, generated, reference, numeric, contract, method, args.dataset, args.catalog, args.seed, args.conditional_samples)
            add_result(result_rows, generated[valid].copy(), reference, numeric, contract, method + "_accepted", args.dataset, args.catalog, args.seed, args.conditional_samples)
            certified, edit_rate, numeric_distortion = certify(generated, contract, heldout)
            add_result(result_rows, certified, reference, numeric, contract, method + "_certified", args.dataset, args.catalog, args.seed,
                       args.conditional_samples, edit_rate, numeric_distortion)
            probability = float(valid.mean())
            for budget in budgets:
                budget_rows.append({
                    "dataset": args.dataset,
                    "catalog": args.catalog,
                    "seed": args.seed,
                    "contract_id": contract["contract_id"],
                    "split": contract["split"],
                    "method": method,
                    "budget": budget,
                    "single_trajectory_success": probability,
                    "success_with_budget": 1.0 - (1.0 - probability) ** budget,
                    "expected_trajectories_per_valid": math.inf if probability == 0 else 1.0 / probability,
                    "heldout_support": float(len(reference) / len(heldout)),
                })

    results = pd.DataFrame(result_rows)
    budgets_frame = pd.DataFrame(budget_rows)
    results["run_label"] = args.run_label
    budgets_frame["run_label"] = args.run_label
    results.to_csv(args.output / "frontier_metrics.csv", index=False)
    budgets_frame.to_csv(args.output / "budget_curve.csv", index=False)
    summary = results.groupby("method").agg(
        contracts=("contract_id", "nunique"), satisfaction=("satisfaction", "mean"), numeric_ks=("numeric_ks", "mean"),
        categorical_tv=("categorical_tv", "mean"), edit_rate=("edit_rate", "mean"), numeric_distortion=("numeric_distortion", "mean")
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "status": "executed",
        "dataset": args.dataset,
        "catalog": args.catalog,
        "seed": args.seed,
        "pool_samples": args.pool_samples,
        "conditional_samples_per_contract": args.conditional_samples,
        "reference": "held-out test cohort",
        "scope": "controlled cohort requests; rejection is a direct baseline",
        "run_label": args.run_label,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
