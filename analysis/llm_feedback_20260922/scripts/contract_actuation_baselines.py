#!/usr/bin/env python3
"""Matched-checkpoint final-repair and best-of-N contract baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditioned_tabdiff_adapter import contract_mask, metrics_for, seed_all
from contract_condition_catalog import catalog
from tabdiff_probe import build_model, decode_tensor


def shortfall(frame: pd.DataFrame, contract: dict, scales: dict[str, float]) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    for predicate in contract["predicates"]:
        values = pd.to_numeric(frame[predicate["column"]], errors="coerce").to_numpy()
        result += np.maximum(float(predicate["threshold"]) - values, 0) / scales[predicate["column"]]
    return result


def final_repair(frame: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, np.ndarray]:
    repaired = frame.copy()
    edited = np.zeros(len(frame), dtype=bool)
    for predicate in contract["predicates"]:
        column, threshold = predicate["column"], float(predicate["threshold"])
        values = pd.to_numeric(repaired[column], errors="coerce")
        mask = values.lt(threshold)
        repaired.loc[mask, column] = threshold
        edited |= mask.to_numpy()
    return repaired, edited


def select_best(candidates: pd.DataFrame, contract: dict, n: int, width: int, scales: dict[str, float]) -> pd.DataFrame:
    chosen = []
    for index in range(n):
        group = candidates.iloc[index * width:(index + 1) * width]
        valid = contract_mask(group, contract).to_numpy()
        if valid.any():
            offset = int(np.flatnonzero(valid)[0])
        else:
            offset = int(shortfall(group, contract, scales).argmin())
        chosen.append(group.iloc[offset])
    return pd.DataFrame(chosen).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--best-of", default="2,4")
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("actuation baselines require GPU")
    device = torch.device("cuda")
    seed_all(args.seed)
    bundle = build_model(args.repo, args.checkpoint, args.config, args.steps, device)
    real = decode_tensor(bundle, bundle.dataset_train.X)
    for column in real.select_dtypes(include=["object", "string"]).columns:
        real[column] = real[column].astype(str).str.strip()
    scales = {column: max(float(pd.to_numeric(real[column]).std()), 1.0) for column in ("age", "education.num", "hours.per.week")}
    widths = [int(value) for value in args.best_of.split(",") if value]
    rows = []
    started = time.perf_counter()
    for ordinal, contract in enumerate(catalog()):
        seed_all(args.seed + 10000 + ordinal)
        with torch.no_grad():
            base = decode_tensor(bundle, bundle.diffusion.sample(args.samples))
        for column in base.select_dtypes(include=["object", "string"]).columns:
            base[column] = base[column].astype(str).str.strip()
        baseline_rate = float(contract_mask(base, contract).mean())
        base_metrics = metrics_for(base, real, contract, baseline_rate)
        base_metrics.update(
            {"method": "unconditional", "seed": args.seed, "denoiser_passes_per_output": 1,
             "edit_rate": 0.0, "target_distortion_std_units": 0.0}
        )
        rows.append(base_metrics)

        repaired, edited = final_repair(base, contract)
        repair_metrics = metrics_for(repaired, real, contract, baseline_rate)
        distortion = np.mean(
            [
                np.abs(pd.to_numeric(repaired[p["column"]]) - pd.to_numeric(base[p["column"]])).mean() / scales[p["column"]]
                for p in contract["predicates"]
            ]
        )
        repair_metrics.update(
            {"method": "final_projection", "seed": args.seed, "denoiser_passes_per_output": 1,
             "edit_rate": float(edited.mean()), "target_distortion_std_units": float(distortion)}
        )
        rows.append(repair_metrics)

        for width in widths:
            seed_all(args.seed + 20000 + ordinal * 31 + width)
            with torch.no_grad():
                pool = decode_tensor(bundle, bundle.diffusion.sample(args.samples * width))
            for column in pool.select_dtypes(include=["object", "string"]).columns:
                pool[column] = pool[column].astype(str).str.strip()
            selected = select_best(pool, contract, args.samples, width, scales)
            selected_metrics = metrics_for(selected, real, contract, baseline_rate)
            selected_metrics.update(
                {"method": f"best_of_{width}", "seed": args.seed, "denoiser_passes_per_output": width,
                 "edit_rate": 0.0, "target_distortion_std_units": None}
            )
            rows.append(selected_metrics)
        print(json.dumps({"contract": contract["contract_id"], "baseline": baseline_rate, "repair_edit_rate": float(edited.mean())}), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "contract_metrics.csv", index=False)
    summary = (
        frame.groupby(["method", "split"])
        .agg(
            contracts=("contract_id", "size"), satisfaction=("satisfaction_rate", "mean"),
            satisfaction_gain=("satisfaction_gain", "mean"), cohort_ks=("cohort_numeric_ks_mean", "mean"),
            categorical_tv=("cohort_categorical_tv_mean", "mean"), coverage=("cohort_category_coverage_mean", "mean"),
            edit_rate=("edit_rate", "mean"), target_distortion=("target_distortion_std_units", "mean"),
            denoiser_passes=("denoiser_passes_per_output", "mean"),
        ).reset_index()
    )
    summary.to_csv(args.output / "summary_by_split.csv", index=False)
    metadata = {
        "status": "executed", "scope": "proxy baselines, not claimed paper reproductions",
        "seed": args.seed, "samples_per_contract": args.samples, "steps": args.steps,
        "best_of": widths, "runtime_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(device), "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "real_data_policy": "Real train rows used locally for aggregate metrics only; no real row written or sent externally.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
