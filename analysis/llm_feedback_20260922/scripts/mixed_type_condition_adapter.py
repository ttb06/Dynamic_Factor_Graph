#!/usr/bin/env python3
"""Mixed numeric/categorical open-contract adapter pilot.

Train only on atomic requests. Evaluate held-out numeric parameters and unseen
categorical/cross-type conjunctions. Contracts are controlled cohort requests,
not mined invariants. The request-ID representation is a negative control for
zero-shot contract composition.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader, TensorDataset

from conditioned_tabdiff_adapter import (
    ConditionedUniModMLP,
    ContractConditionedModel,
    decode,
    seed_all,
    total_variation,
)
from tabdiff_probe import preprocess_dataset


SPECS = {
    "adult": {
        "numeric": {
            "age": {"scale": 100.0, "train": [25, 35, 50], "heldout": [30, 45]},
            "hours.per.week": {"scale": 100.0, "train": [20, 40, 50], "heldout": [30, 45]},
            "education.num": {"scale": 20.0, "train": [9, 10, 13], "heldout": [11, 12]},
        },
        "categorical": {
            "workclass": ["Private", "Self-emp-not-inc"],
            "education": ["HS-grad", "Bachelors", "Some-college"],
            "occupation": ["Exec-managerial", "Sales"],
        },
        "categorical_composition": [
            [("workclass", "eq", "Private"), ("education", "eq", "HS-grad")],
            [("education", "eq", "Bachelors"), ("occupation", "eq", "Exec-managerial")],
            [("workclass", "eq", "Self-emp-not-inc"), ("occupation", "eq", "Sales")],
        ],
        "mixed_composition": [
            [("age", "ge", 35.0), ("workclass", "eq", "Private")],
            [("hours.per.week", "ge", 40.0), ("education", "eq", "Bachelors")],
            [("education.num", "ge", 10.0), ("occupation", "eq", "Exec-managerial")],
        ],
    },
    "default": {
        "numeric": {
            "LIMIT_BAL": {"scale": 1_000_000.0, "train": [50_000, 140_000, 360_000], "heldout": [100_000, 250_000]},
            "AGE": {"scale": 100.0, "train": [25, 34, 49], "heldout": [30, 45]},
            "BILL_AMT1": {"scale": 1_000_000.0, "train": [0, 22_000, 140_000], "heldout": [10_000, 70_000]},
            # Payment columns support the subgroup-graph experiment.  The
            # adapter is trained only on atomic threshold requests; row-wise
            # subgroup composition remains held out.
            "PAY_AMT3": {"scale": 100_000.0, "train": [1, 1_000, 5_000], "heldout": [500, 2_000]},
            "PAY_AMT4": {"scale": 100_000.0, "train": [1, 1_000, 5_000], "heldout": [500, 2_000]},
            "PAY_AMT5": {"scale": 100_000.0, "train": [1, 1_000, 5_000], "heldout": [500, 2_000]},
        },
        "categorical": {"SEX": ["1", "2"], "EDUCATION": ["1", "2", "3"], "MARRIAGE": ["1", "2"]},
        "categorical_composition": [
            [("SEX", "eq", "2"), ("EDUCATION", "eq", "2")],
            [("SEX", "eq", "1"), ("MARRIAGE", "eq", "1")],
            [("EDUCATION", "eq", "1"), ("MARRIAGE", "eq", "2")],
        ],
        "mixed_composition": [
            [("AGE", "ge", 34.0), ("MARRIAGE", "eq", "1")],
            [("LIMIT_BAL", "ge", 140_000.0), ("EDUCATION", "eq", "2")],
            [("BILL_AMT1", "ge", 22_000.0), ("SEX", "eq", "2")],
        ],
    },
    "shoppers": {
        "numeric": {
            "PageValues": {"scale": 400.0, "train": [1, 10, 40], "heldout": [5, 25]},
            "Administrative_Duration": {"scale": 3_500.0, "train": [10, 90, 220], "heldout": [50, 150]},
            "ProductRelated_Duration": {"scale": 65_000.0, "train": [200, 600, 2_800], "heldout": [400, 1_500]},
        },
        "categorical": {
            "Month": ["May", "Nov", "Mar", "Dec"],
            "VisitorType": ["Returning_Visitor", "New_Visitor"],
            "Weekend": ["False", "True"],
        },
        "categorical_composition": [
            [("VisitorType", "eq", "New_Visitor"), ("Weekend", "eq", "False")],
            [("Month", "eq", "Nov"), ("VisitorType", "eq", "Returning_Visitor")],
            [("Month", "eq", "May"), ("Weekend", "eq", "True")],
        ],
        "mixed_composition": [
            [("PageValues", "ge", 10.0), ("VisitorType", "eq", "New_Visitor")],
            [("Administrative_Duration", "ge", 90.0), ("Weekend", "eq", "True")],
            [("ProductRelated_Duration", "ge", 600.0), ("Month", "eq", "Nov")],
        ],
    },
    "diabetes": {
        "numeric": {
            "time_in_hospital": {"scale": 14.0, "train": [2, 4, 9], "heldout": [3, 7]},
            "num_medications": {"scale": 100.0, "train": [10, 15, 26], "heldout": [12, 22]},
            "num_lab_procedures": {"scale": 150.0, "train": [31, 44, 67], "heldout": [37, 60]},
        },
        "categorical": {
            "change": ["No", "Ch"],
            "diabetesMed": ["Yes", "No"],
            "insulin": ["No", "Steady", "Up", "Down"],
        },
        "categorical_composition": [
            [("change", "eq", "Ch"), ("insulin", "eq", "Up")],
            [("diabetesMed", "eq", "Yes"), ("insulin", "eq", "Steady")],
            [("change", "eq", "No"), ("insulin", "eq", "No")],
        ],
        "mixed_composition": [
            [("time_in_hospital", "ge", 4.0), ("change", "eq", "Ch")],
            [("num_medications", "ge", 15.0), ("diabetesMed", "eq", "Yes")],
            [("num_lab_procedures", "ge", 44.0), ("insulin", "eq", "Steady")],
        ],
    },
}


def predicate_id(predicate: tuple[str, str, object]) -> str:
    column, operator, value = predicate
    return f"{column}_{operator}_{value}"


def make_catalog(dataset: str) -> list[dict]:
    spec = SPECS[dataset]
    contracts = []
    for column, entry in spec["numeric"].items():
        for split, key in (("train", "train"), ("heldout_threshold", "heldout")):
            for value in entry[key]:
                predicate = (column, "ge", float(value))
                contracts.append({"contract_id": predicate_id(predicate), "split": split, "family": column, "predicates": [predicate]})
    for column, values in spec["categorical"].items():
        for value in values:
            predicate = (column, "eq", value)
            contracts.append({"contract_id": predicate_id(predicate), "split": "train", "family": column, "predicates": [predicate]})
    for split in ("categorical_composition", "mixed_composition"):
        for ordinal, predicates in enumerate(spec[split]):
            contracts.append({
                "contract_id": "and_" + "__".join(predicate_id(predicate) for predicate in predicates),
                "split": f"heldout_{split}", "family": f"{split}_{ordinal}", "predicates": predicates,
            })
    return contracts


def structured_vector(contract: dict, dataset: str) -> list[float]:
    spec = SPECS[dataset]
    vector = []
    predicates = {(column, operator): value for column, operator, value in contract["predicates"]}
    for column, entry in spec["numeric"].items():
        value = predicates.get((column, "ge"))
        vector.extend([float(value is not None), 0.0 if value is None else float(value) / entry["scale"]])
    for column, values in spec["categorical"].items():
        value = predicates.get((column, "eq"))
        vector.append(float(value is not None))
        vector.extend(float(value == candidate) for candidate in values)
    return vector


def mask(frame: pd.DataFrame, contract: dict) -> pd.Series:
    result = pd.Series(True, index=frame.index)
    for column, operator, value in contract["predicates"]:
        if operator == "ge":
            result &= pd.to_numeric(frame[column], errors="coerce").ge(float(value))
        elif operator == "eq":
            result &= frame[column].astype(str).eq(str(value))
        else:
            raise ValueError(operator)
    return result


def evaluate(sample: pd.DataFrame, real: pd.DataFrame, contract: dict) -> dict:
    selected = mask(sample, contract)
    reference = real[mask(real, contract)]
    numeric = list(NUMERIC_COLUMNS)
    categorical = [column for column in real.columns if column not in numeric]
    target_columns = {column for column, _, _ in contract["predicates"]}
    ks_by_column = {column: float(ks_2samp(pd.to_numeric(sample[column]), pd.to_numeric(reference[column])).statistic) for column in numeric}
    tv_by_column = {column: total_variation(sample[column], reference[column]) for column in categorical}
    target_numeric = [value for column, value in ks_by_column.items() if column in target_columns]
    nontarget_numeric = [value for column, value in ks_by_column.items() if column not in target_columns]
    target_categorical = [value for column, value in tv_by_column.items() if column in target_columns]
    nontarget_categorical = [value for column, value in tv_by_column.items() if column not in target_columns]
    return {
        "contract_id": contract["contract_id"], "split": contract["split"], "family": contract["family"],
        "satisfaction_rate": float(selected.mean()), "cohort_numeric_ks": float(np.mean(list(ks_by_column.values()))),
        "cohort_categorical_tv": float(np.mean(list(tv_by_column.values()))),
        "target_numeric_ks": float(np.mean(target_numeric)) if target_numeric else np.nan,
        "nontarget_numeric_ks": float(np.mean(nontarget_numeric)) if nontarget_numeric else np.nan,
        "target_categorical_tv": float(np.mean(target_categorical)) if target_categorical else np.nan,
        "nontarget_categorical_tv": float(np.mean(nontarget_categorical)) if nontarget_categorical else np.nan,
        "reference_cohort_size": len(reference),
    }


def exact_project(sample: pd.DataFrame, real: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, float, float, float]:
    projected = sample.copy()
    edited = np.zeros(len(projected), dtype=bool)
    numeric_distortions = []
    categorical_edits = np.zeros(len(projected), dtype=bool)
    for column, operator, value in contract["predicates"]:
        if operator == "ge":
            old = pd.to_numeric(projected[column]).copy()
            bad = old.lt(float(value))
            projected.loc[bad, column] = float(value)
            scale = max(float(pd.to_numeric(real[column]).std()), 1e-6)
            numeric_distortions.append(float((pd.to_numeric(projected[column]) - old).abs().mean() / scale))
        else:
            bad = ~projected[column].astype(str).eq(str(value))
            projected.loc[bad, column] = str(value)
            categorical_edits |= bad.to_numpy()
        edited |= bad.to_numpy()
    numeric_distortion = float(np.mean(numeric_distortions)) if numeric_distortions else 0.0
    return projected, float(edited.mean()), numeric_distortion, float(categorical_edits.mean())


NUMERIC_COLUMNS: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", choices=["adult", "default", "shoppers", "diabetes"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=["request_id", "structured", "text", "hybrid"], required=True)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pairs-per-contract", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--adapter-lr", type=float, default=5e-4)
    parser.add_argument("--core-lr", type=float, default=1e-5)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--threshold-curriculum", choices=["fixed", "dense"], default="fixed")
    parser.add_argument("--dense-quantiles", default="0.1:0.2:0.3:0.4:0.5:0.6:0.7:0.8:0.9")
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--guidance-scales", default="1:2:4")
    parser.add_argument("--run-label", default="unspecified")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    started = time.perf_counter()
    device = torch.device("cuda")
    seed_all(args.seed)
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    contracts = make_catalog(args.dataset)
    train_contracts = [contract for contract in contracts if contract["split"] == "train"]
    train_ids = [contract["contract_id"] for contract in train_contracts]
    text_vectors = {}
    embedding_meta = {}
    if args.representation in {"text", "hybrid"}:
        if args.embeddings is None:
            raise ValueError("text/hybrid representation requires --embeddings")
        embedding_payload = json.loads(args.embeddings.read_text())
        text_vectors = embedding_payload["vectors"]
        embedding_meta = {
            key: embedding_payload.get(key) for key in ("model", "dimensions", "usage", "factorized")
        }
    vectors = {}
    for contract in contracts:
        if args.representation == "structured":
            vector = structured_vector(contract, args.dataset)
        elif args.representation == "request_id":
            vector = [0.0] * len(train_ids)
            if contract["contract_id"] in train_ids:
                vector[train_ids.index(contract["contract_id"])] = 1.0
        elif args.representation == "text":
            vector = list(text_vectors[contract["contract_id"]])
        elif args.representation == "hybrid":
            vector = structured_vector(contract, args.dataset) + list(text_vectors[contract["contract_id"]])
        else:
            raise ValueError(args.representation)
        array = np.asarray(vector, dtype=np.float32)
        if args.representation in {"text", "hybrid"}:
            prefix = len(structured_vector(contract, args.dataset)) if args.representation == "hybrid" else 0
            norm = np.linalg.norm(array[prefix:])
            if norm > 0:
                array[prefix:] /= norm
        vectors[contract["contract_id"]] = array.tolist()
    condition_dim = len(next(iter(vectors.values())))

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, _ = preprocess_dataset(args.repo, args.dataset)
    real = decode(train_data, train_data.X, info)
    global NUMERIC_COLUMNS
    NUMERIC_COLUMNS = [info["column_names"][index] for index in info["num_col_idx"]]
    if args.threshold_curriculum == "dense":
        if args.representation != "structured":
            raise ValueError("dense threshold curriculum currently requires the structured representation")
        known_ids = {contract["contract_id"] for contract in contracts}
        quantiles = [float(value) for value in args.dense_quantiles.replace(",", ":").split(":") if value]
        for column in SPECS[args.dataset]["numeric"]:
            values = pd.to_numeric(real[column], errors="coerce")
            for quantile in quantiles:
                threshold = float(values.quantile(quantile))
                predicate = (column, "ge", threshold)
                contract_id = "dense_" + predicate_id(predicate)
                if contract_id in known_ids:
                    continue
                contract = {
                    "contract_id": contract_id, "split": "train", "family": column,
                    "predicates": [predicate], "training_quantile": quantile,
                }
                contracts.append(contract)
                train_contracts.append(contract)
                vectors[contract_id] = structured_vector(contract, args.dataset)
                known_ids.add(contract_id)
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    backbone = ConditionedUniModMLP(condition_dim=condition_dim, **params)
    base = Model(backbone, **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = base.load_state_dict(checkpoint["denoise_fn"], strict=False)
    missing = [key for key in incompatible.missing_keys if "condition_projection" not in key]
    if incompatible.unexpected_keys or missing:
        raise RuntimeError(str(incompatible))
    model = ContractConditionedModel(base, backbone, condition_dim).to(device)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, model, None, **diffusion_params, device=device).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    for parameter in diffusion.parameters():
        parameter.requires_grad_(False)
    for parameter in backbone.mlp.condition_projection.parameters():
        parameter.requires_grad_(True)
    for parameter in backbone.mlp.mlp.parameters():
        parameter.requires_grad_(args.core_lr > 0)

    rng = np.random.default_rng(args.seed)
    examples, conditions, eligible_counts = [], [], {}
    for contract in train_contracts:
        eligible = np.flatnonzero(mask(real, contract).to_numpy())
        if not len(eligible):
            raise RuntimeError(f"no eligible rows for {contract['contract_id']}")
        indices = rng.choice(eligible, args.pairs_per_contract, replace=len(eligible) < args.pairs_per_contract)
        examples.append(train_data.X[indices])
        conditions.append(torch.tensor(vectors[contract["contract_id"]], dtype=torch.float32).repeat(args.pairs_per_contract, 1))
        eligible_counts[contract["contract_id"]] = len(eligible)
    indices = rng.choice(len(real), args.pairs_per_contract, replace=False)
    examples.append(train_data.X[indices])
    conditions.append(torch.zeros(args.pairs_per_contract, condition_dim))
    all_examples, all_conditions = torch.cat(examples), torch.cat(conditions)
    loader = DataLoader(TensorDataset(all_examples, all_conditions), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), pin_memory=True)
    parameter_groups = [{"params": backbone.mlp.condition_projection.parameters(), "lr": args.adapter_lr}]
    if args.core_lr > 0:
        parameter_groups.append({"params": backbone.mlp.mlp.parameters(), "lr": args.core_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0)
    curve = []
    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        discrete_sum = continuous_sum = count = 0.0
        for batch, condition in loader:
            batch, condition = batch.to(device), condition.to(device)
            dropped = torch.rand(len(condition), device=device) < args.condition_dropout
            condition = condition.clone(); condition[dropped] = 0
            model.set_condition(condition, 1.0)
            optimizer.zero_grad(set_to_none=True)
            continuous_loss, discrete_loss = diffusion.mixed_loss(batch)
            (continuous_loss + discrete_loss).backward(); optimizer.step()
            discrete_sum += float(discrete_loss.detach()) * len(batch)
            continuous_sum += float(continuous_loss.detach()) * len(batch)
            count += len(batch)
        curve.append({"epoch": epoch, "discrete_loss": discrete_sum / count, "continuous_loss": continuous_sum / count})
    pd.DataFrame(curve).to_csv(args.output / "training_curve.csv", index=False)

    diffusion.eval()
    heldout = [contract for contract in contracts if contract["split"] != "train"]
    model.set_condition(torch.zeros(args.eval_samples, condition_dim, device=device), 1.0)
    seed_all(args.seed + 9000)
    with torch.no_grad():
        unconditional = decode(train_data, diffusion.sample(args.eval_samples), info)
    rows = []
    for contract in heldout:
        metric = evaluate(unconditional, real, contract)
        metric.update(dataset=args.dataset, representation=args.representation, method="unconditional", guidance_scale=0.0,
                      seed=args.seed, edit_rate=0.0, numeric_target_distortion=0.0, categorical_target_edit_rate=0.0)
        rows.append(metric)
        if args.representation == "structured":
            projected, edit_rate, distortion, categorical_edit = exact_project(unconditional, real, contract)
            metric = evaluate(projected, real, contract)
            metric.update(dataset=args.dataset, representation=args.representation, method="unconditional_then_final_projection",
                          guidance_scale=0.0, seed=args.seed, edit_rate=edit_rate,
                          numeric_target_distortion=distortion, categorical_target_edit_rate=categorical_edit)
            rows.append(metric)
    for guidance in [float(value) for value in args.guidance_scales.replace(",", ":").split(":") if value]:
        for ordinal, contract in enumerate(heldout):
            condition = torch.tensor(vectors[contract["contract_id"]], dtype=torch.float32, device=device).repeat(args.eval_samples, 1)
            model.set_condition(condition, guidance)
            seed_all(args.seed + 10000 + ordinal)
            with torch.no_grad():
                sample = decode(train_data, diffusion.sample(args.eval_samples), info)
            metric = evaluate(sample, real, contract)
            metric.update(dataset=args.dataset, representation=args.representation, method=f"{args.representation}_condition",
                          guidance_scale=guidance, seed=args.seed, edit_rate=0.0, numeric_target_distortion=0.0,
                          categorical_target_edit_rate=0.0)
            rows.append(metric)
            if args.representation == "structured":
                projected, edit_rate, distortion, categorical_edit = exact_project(sample, real, contract)
                metric = evaluate(projected, real, contract)
                metric.update(dataset=args.dataset, representation=args.representation, method="structured_then_final_projection",
                              guidance_scale=guidance, seed=args.seed, edit_rate=edit_rate,
                              numeric_target_distortion=distortion, categorical_target_edit_rate=categorical_edit)
                rows.append(metric)
    metrics = pd.DataFrame(rows)
    metrics["run_label"] = args.run_label
    metrics.to_csv(args.output / "contract_metrics.csv", index=False)
    summary = metrics.groupby(["method", "guidance_scale", "split"]).agg(
        contracts=("contract_id", "size"), satisfaction=("satisfaction_rate", "mean"),
        cohort_ks=("cohort_numeric_ks", "mean"), categorical_tv=("cohort_categorical_tv", "mean"),
        target_numeric_ks=("target_numeric_ks", "mean"), nontarget_numeric_ks=("nontarget_numeric_ks", "mean"),
        target_categorical_tv=("target_categorical_tv", "mean"), nontarget_categorical_tv=("nontarget_categorical_tv", "mean"),
        edit_rate=("edit_rate", "mean"), numeric_target_distortion=("numeric_target_distortion", "mean"),
        categorical_target_edit_rate=("categorical_target_edit_rate", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    torch.save({"condition_projection": backbone.mlp.condition_projection.state_dict(), "latent_mlp": backbone.mlp.mlp.state_dict(),
                "vectors": vectors, "condition_dim": condition_dim}, args.output / "adapter.pt")
    metadata = {
        "status": "executed", "dataset": args.dataset, "representation": args.representation, "seed": args.seed,
        "epochs": args.epochs, "training_pairs": len(all_examples), "samples_per_contract": args.eval_samples,
        "adapter_lr": args.adapter_lr, "core_lr": args.core_lr, "condition_dropout": args.condition_dropout,
        "threshold_curriculum": args.threshold_curriculum,
        "run_label": args.run_label,
        "embedding": embedding_meta,
        "eligible_counts": eligible_counts, "runtime_seconds": time.perf_counter() - started,
        "scope": "controlled mixed-type cohort requests; atomic training, held-out composition",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
