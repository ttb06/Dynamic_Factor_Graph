#!/usr/bin/env python3
"""Approximate the oracle active set with calibrated early row evidence.

The only population-level side information is applicability prevalence on the
real training split. Evaluation-row applicability is never exposed. All
methods reuse the same initial noise and only actuate the numeric consequent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from conditioned_tabdiff_adapter import decode, seed_all
from delayed_gate_schedule_probe import build
from implication_condition_adapter import clean, evaluate
from implication_contract_catalog import antecedent_mask, catalog, exception_mask, vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--gate-start", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")

    device = torch.device("cuda")
    seed_all(args.seed)
    started = time.perf_counter()
    contracts, info, train_data, conditioned, diffusion = build(args, device)
    real = clean(decode(train_data, train_data.X, info))
    conditioned.actuation_scope = "target"

    train_priors = {}
    for contract in contracts:
        active = antecedent_mask(real, contract) & ~exception_mask(real, contract)
        train_priors[contract["contract_id"]] = float(active.mean())

    rows, baselines, preview_gates = [], {}, {}
    conditioned.early_policy = "off"
    conditioned.gate_start_t = -1
    for ordinal, contract in enumerate(contracts):
        condition = torch.tensor(vector(contract), dtype=torch.float32, device=device).repeat(args.samples, 1)
        conditioned.set_condition(condition, args.guidance)
        seed_all(args.seed + 10000 + ordinal)
        with torch.no_grad():
            frame = clean(decode(train_data, diffusion.sample(args.samples), info))
        metric = evaluate(frame, real, contract)
        metric.update(
            seed=args.seed, guidance_scale=args.guidance, gate_start_t=-1,
            schedule="no_actuation", early_policy="off", gate_power=1.0,
            train_applicability_prior=train_priors[contract["contract_id"]],
        )
        rows.append(metric)
        baselines[contract["contract_id"]] = metric
        active = antecedent_mask(frame, contract) & ~exception_mask(frame, contract)
        preview_gates[contract["contract_id"]] = torch.tensor(
            active.to_numpy(), dtype=torch.float32, device=device
        )

    specs = [
        ("uncalibrated_soft", "soft", 1.0),
        ("calibrated_soft_power_0.25", "calibrated_soft", 0.25),
        ("calibrated_soft_power_0.5", "calibrated_soft", 0.5),
        ("calibrated_soft_power_1", "calibrated_soft", 1.0),
        ("calibrated_soft_power_2", "calibrated_soft", 2.0),
        ("prevalence_topk", "topk_soft", 1.0),
        ("paired_unconditional_preview_rerun", "external", 1.0),
    ]
    conditioned.gate_start_t = args.gate_start
    for schedule, policy, power in specs:
        conditioned.early_policy = policy
        conditioned.early_floor = 0.0
        conditioned.gate_power = power
        for ordinal, contract in enumerate(contracts):
            contract_id = contract["contract_id"]
            conditioned.external_prior = train_priors[contract_id]
            conditioned.external_gate = preview_gates[contract_id] if policy == "external" else None
            condition = torch.tensor(vector(contract), dtype=torch.float32, device=device).repeat(args.samples, 1)
            conditioned.set_condition(condition, args.guidance)
            seed_all(args.seed + 10000 + ordinal)
            with torch.no_grad():
                frame = clean(decode(train_data, diffusion.sample(args.samples), info))
            metric = evaluate(frame, real, contract, baselines[contract_id])
            metric.update(
                seed=args.seed, guidance_scale=args.guidance, gate_start_t=args.gate_start,
                schedule=schedule, early_policy=policy, gate_power=power,
                train_applicability_prior=train_priors[contract_id],
            )
            rows.append(metric)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "active_set_metrics.csv", index=False)
    summary = metrics.groupby(
        ["schedule", "early_policy", "gate_power", "split"], dropna=False
    ).agg(
        contracts=("contract_id", "size"),
        violation=("conditional_violation_rate", "mean"),
        reduction=("conditional_violation_reduction", "mean"),
        antecedent_error=("antecedent_rate_error", "mean"),
        exception_error=("exception_rate_error", "mean"),
        inactive_error=("inactive_target_mean_error", "mean"),
        numeric_ks=("global_numeric_ks", "mean"),
        categorical_tv=("global_categorical_tv", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "status": "executed",
        "seed": args.seed,
        "guidance": args.guidance,
        "gate_start": args.gate_start,
        "samples_per_contract": args.samples,
        "steps": args.steps,
        "population_information": "train-split applicability prevalence per contract",
        "preview_definition": (
            "Applicability of a paired unconditional final sample from the same initial noise. "
            "Operational two-pass method under target-scoped actuation, not a ground-truth oracle."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
