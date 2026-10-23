#!/usr/bin/env python3
"""Smoke/fidelity/rule audit for newly trained Default and Shoppers checkpoints."""

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

from conditioned_tabdiff_adapter import decode, seed_all, total_variation
from tabdiff_probe import preprocess_dataset


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.select_dtypes(include=["object", "string"]).columns:
        out[column] = out[column].astype(str).str.strip()
    return out


def numeric(frame, column):
    return pd.to_numeric(frame[column], errors="coerce")


def rule_specs(dataset: str):
    if dataset == "shoppers":
        specs = []
        for prefix in ("Administrative", "Informational", "ProductRelated"):
            duration = f"{prefix}_Duration"
            specs.append((f"{prefix}=0 -> {duration}=0", lambda f, p=prefix, d=duration: numeric(f, p).eq(0),
                          lambda f, d=duration: numeric(f, d).abs().le(1e-8)))
            specs.append((f"{duration}=0 -> {prefix}=0", lambda f, d=duration: numeric(f, d).abs().le(1e-8),
                          lambda f, p=prefix: numeric(f, p).eq(0)))
        specs.append(("BounceRates <= ExitRates", lambda f: pd.Series(True, index=f.index),
                      lambda f: numeric(f, "BounceRates").le(numeric(f, "ExitRates") + 1e-9)))
        return specs
    return [
        ("PAY_0=0 -> BILL_AMT1>0", lambda f: f["PAY_0"].astype(str).eq("0"), lambda f: numeric(f, "BILL_AMT1").gt(0)),
        ("PAY_0=-1 -> BILL_AMT1>0", lambda f: f["PAY_0"].astype(str).eq("-1"), lambda f: numeric(f, "BILL_AMT1").gt(0)),
        ("PAY_3=0 -> BILL_AMT2>0", lambda f: f["PAY_3"].astype(str).eq("0"), lambda f: numeric(f, "BILL_AMT2").gt(0)),
        ("PAY_5=0 -> BILL_AMT4>0", lambda f: f["PAY_5"].astype(str).eq("0"), lambda f: numeric(f, "BILL_AMT4").gt(0)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", choices=["default", "shoppers"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    device = torch.device("cuda"); seed_all(args.seed); sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    started = time.perf_counter()
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, args.dataset)
    with args.config.open("rb") as handle: config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical; params["categories"] = (train_data.categories + 1).tolist()
    backbone = UniModMLP(**params); model = Model(backbone, **config["diffusion_params"]["edm_params"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["denoise_fn"])
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, model, None,
                                      **diffusion_params, device=device).to(device)
    diffusion.num_schedule.load_state_dict(state["num_schedule"]); diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval(); seed_all(args.seed)
    with torch.no_grad(): encoded = diffusion.sample(args.samples)
    synthetic = clean(decode(train_data, encoded, info)); train = clean(decode(train_data, train_data.X, info)); test = clean(decode(train_data, test_data.X, info))
    synthetic.insert(0, "privacy", "synthetic_generated")
    synthetic.to_csv(args.output / "synthetic_samples.csv", index=False)
    synthetic_eval = synthetic.drop(columns="privacy")

    num_names = [info["column_names"][i] for i in info["num_col_idx"]]
    cat_names = [info["column_names"][i] for i in info["cat_col_idx"] + info["target_col_idx"]]
    fidelity = []
    for reference_name, reference in (("train", train), ("test", test)):
        for column in num_names:
            fidelity.append({"reference": reference_name, "column": column, "type": "numeric_ks",
                             "value": float(ks_2samp(numeric(synthetic_eval, column), numeric(reference, column)).statistic)})
        for column in cat_names:
            fidelity.append({"reference": reference_name, "column": column, "type": "categorical_tv",
                             "value": total_variation(synthetic_eval[column], reference[column])})
    pd.DataFrame(fidelity).to_csv(args.output / "fidelity_by_column.csv", index=False)

    rule_rows = []
    for name, antecedent_fn, consequent_fn in rule_specs(args.dataset):
        for split, frame in (("train", train), ("test", test), ("synthetic", synthetic_eval)):
            antecedent = antecedent_fn(frame).fillna(False); consequent = consequent_fn(frame).fillna(False)
            applicable = int(antecedent.sum()); violations = int((antecedent & ~consequent).sum())
            rule_rows.append({"rule": name, "split": split, "rows": len(frame), "applicable": applicable,
                              "coverage": applicable / len(frame), "violations": violations,
                              "violation_rate": violations / max(applicable, 1)})
    rules = pd.DataFrame(rule_rows); rules.to_csv(args.output / "rule_audit.csv", index=False)
    fid = pd.DataFrame(fidelity)
    summary = {
        "status": "executed", "dataset": args.dataset, "samples": args.samples, "steps": args.steps,
        "mean_numeric_ks_test": float(fid[(fid.reference == "test") & (fid.type == "numeric_ks")].value.mean()),
        "mean_categorical_tv_test": float(fid[(fid.reference == "test") & (fid.type == "categorical_tv")].value.mean()),
        "duplicate_rate": float(synthetic_eval.duplicated().mean()), "runtime_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(device),
        "real_data_policy": "Real rows used locally for metrics; only synthetic rows and aggregate metrics are written.",
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(rules.to_string(index=False)); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
