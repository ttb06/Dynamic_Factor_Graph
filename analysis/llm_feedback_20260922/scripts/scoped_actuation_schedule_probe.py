#!/usr/bin/env python3
"""Test whether a typed target-column actuator cleans up dynamic guidance.

The same condition adapter, contracts, initial noise, and row gate are used for
both scopes.  ``all`` exposes the backbone's full conditional delta, whereas
``target`` retains only the requested numeric consequent and leaves categorical
logits unconditional.
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
from implication_contract_catalog import vector


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

    specs = [
        ("never", "all", "off", -1.0),
        ("always_all", "all", "gated", 1.0),
        ("always_target", "target", "gated", 1.0),
        ("global_then_gate_0.5_all", "all", "global", 0.5),
        ("global_then_gate_0.5_target", "target", "global", 0.5),
        ("global_then_gate_0.25_all", "all", "global", 0.25),
        ("global_then_gate_0.25_target", "target", "global", 0.25),
        ("global_then_gate_0.1_all", "all", "global", 0.1),
        ("global_then_gate_0.1_target", "target", "global", 0.1),
    ]
    rows = []
    baselines = {}
    for schedule, scope, early_policy, gate_start in specs:
        conditioned.actuation_scope = scope
        conditioned.early_policy = early_policy
        conditioned.gate_start_t = gate_start
        for ordinal, contract in enumerate(contracts):
            condition = torch.tensor(vector(contract), dtype=torch.float32, device=device).repeat(args.samples, 1)
            conditioned.set_condition(condition, args.guidance)
            seed_all(args.seed + 10000 + ordinal)
            with torch.no_grad():
                frame = clean(decode(train_data, diffusion.sample(args.samples), info))
            metric = evaluate(frame, real, contract)
            if schedule == "never":
                baselines[contract["contract_id"]] = metric
            else:
                base = baselines[contract["contract_id"]]
                metric["conditional_violation_reduction"] = base["conditional_violation_rate"] - metric["conditional_violation_rate"]
                metric["antecedent_error_change"] = metric["antecedent_rate_error"] - base["antecedent_rate_error"]
            metric.update(seed=args.seed, guidance_scale=args.guidance, schedule=schedule,
                          actuation_scope=scope, early_policy=early_policy, gate_start_t=gate_start)
            rows.append(metric)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "scoped_metrics.csv", index=False)
    summary = metrics.groupby(["schedule", "actuation_scope", "gate_start_t", "split"]).agg(
        contracts=("contract_id", "size"), violation=("conditional_violation_rate", "mean"),
        reduction=("conditional_violation_reduction", "mean"), antecedent_error=("antecedent_rate_error", "mean"),
        exception_error=("exception_rate_error", "mean"), inactive_error=("inactive_target_mean_error", "mean"),
        numeric_ks=("global_numeric_ks", "mean"), categorical_tv=("global_categorical_tv", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "status": "executed", "seed": args.seed, "samples_per_contract": args.samples,
        "steps": args.steps, "guidance": args.guidance,
        "scope": "paired-noise inference-only target-scoped actuation; no retraining",
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
