#!/usr/bin/env python3
"""Causal controls for row applicability under paired initial noise.

Tests whether early posterior applicability is more useful than global,
shuffled, or oracle-final gates. All conditional deltas are target-scoped.
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
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    device = torch.device("cuda"); seed_all(args.seed); started = time.perf_counter()
    contracts, info, train_data, conditioned, diffusion = build(args, device)
    real = clean(decode(train_data, train_data.X, info))
    conditioned.actuation_scope = "target"

    rows, baselines, oracle_gates = [], {}, {}
    conditioned.early_policy = "off"; conditioned.gate_start_t = -1
    for ordinal, contract in enumerate(contracts):
        condition = torch.tensor(vector(contract), dtype=torch.float32, device=device).repeat(args.samples, 1)
        conditioned.set_condition(condition, args.guidance)
        seed_all(args.seed + 10000 + ordinal)
        with torch.no_grad(): frame = clean(decode(train_data, diffusion.sample(args.samples), info))
        metric = evaluate(frame, real, contract)
        metric.update(seed=args.seed, guidance_scale=args.guidance, gate_start_t=-1,
                      schedule="never", early_policy="off", early_floor=0.0)
        rows.append(metric); baselines[contract["contract_id"]] = metric
        active = antecedent_mask(frame, contract) & ~exception_mask(frame, contract)
        oracle_gates[contract["contract_id"]] = torch.tensor(active.to_numpy(), dtype=torch.float32, device=device)

    specs = [
        ("always_hard", "gated", 1.0, 0.0),
        (f"global_then_hard_{args.gate_start:g}", "global", args.gate_start, 0.0),
        (f"soft0_then_hard_{args.gate_start:g}", "soft", args.gate_start, 0.0),
        (f"soft25_then_hard_{args.gate_start:g}", "soft", args.gate_start, 0.25),
        (f"soft50_then_hard_{args.gate_start:g}", "soft", args.gate_start, 0.50),
        (f"soft75_then_hard_{args.gate_start:g}", "soft", args.gate_start, 0.75),
        (f"shuffled_then_hard_{args.gate_start:g}", "shuffled", args.gate_start, 0.0),
        ("oracle_baseline_applicability", "external", 1.0, 0.0),
    ]
    for schedule, policy, gate_start, floor in specs:
        conditioned.early_policy = policy; conditioned.gate_start_t = gate_start; conditioned.early_floor = floor
        for ordinal, contract in enumerate(contracts):
            conditioned.external_gate = oracle_gates[contract["contract_id"]] if policy == "external" else None
            condition = torch.tensor(vector(contract), dtype=torch.float32, device=device).repeat(args.samples, 1)
            conditioned.set_condition(condition, args.guidance)
            seed_all(args.seed + 10000 + ordinal)
            with torch.no_grad(): frame = clean(decode(train_data, diffusion.sample(args.samples), info))
            metric = evaluate(frame, real, contract, baselines[contract["contract_id"]])
            metric.update(seed=args.seed, guidance_scale=args.guidance, gate_start_t=gate_start,
                          schedule=schedule, early_policy=policy, early_floor=floor)
            rows.append(metric)
    metrics = pd.DataFrame(rows); metrics.to_csv(args.output / "control_metrics.csv", index=False)
    summary = metrics.groupby(["schedule", "early_policy", "early_floor", "split"]).agg(
        contracts=("contract_id", "size"), violation=("conditional_violation_rate", "mean"),
        reduction=("conditional_violation_reduction", "mean"), antecedent_error=("antecedent_rate_error", "mean"),
        exception_error=("exception_rate_error", "mean"), inactive_error=("inactive_target_mean_error", "mean"),
        numeric_ks=("global_numeric_ks", "mean"), categorical_tv=("global_categorical_tv", "mean"),
    ).reset_index(); summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {"status":"executed", "seed":args.seed, "guidance":args.guidance,
                "gate_start":args.gate_start, "samples_per_contract":args.samples, "steps":args.steps,
                "oracle_definition":"Applicability of paired no-actuation final row; diagnostic upper-bound control.",
                "runtime_seconds":time.perf_counter()-started}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__": main()
