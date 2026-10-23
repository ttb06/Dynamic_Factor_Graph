#!/usr/bin/env python3
"""Sweep one-shot and two-shot feedback timing on paired TabDiff seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import tabdiff_probe as core
from persistence_feedback_probe import sample_selective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260922, 20260923, 20260924])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    bundle = core.build_model(args.repo, args.checkpoint, args.config, args.steps, device)
    real = pd.read_csv(args.repo / "synthetic/adult/real.csv")
    test = pd.read_csv(args.repo / "synthetic/adult/test.csv")
    context = core.make_rule_context(real)

    single_times = [0.35, 0.25, 0.17, 0.10, 0.03]
    paired_times = [(0.35, 0.10), (0.25, 0.03), (0.17, 0.03), (0.10, 0.03)]
    rows = []
    for seed in args.seeds:
        for intervention_t in single_times:
            syn, _, runtime, diagnostics, _ = sample_selective(
                bundle,
                context,
                args.samples,
                args.steps,
                seed,
                "late_once",
                intervention_t,
                0.03,
                collect_forecast=False,
            )
            label = f"single_t{intervention_t:.2f}"
            metric = core.evaluate_variant(
                label,
                syn,
                real,
                test,
                context,
                bundle,
                runtime,
                args.steps,
                diagnostics["correction_relation_events"],
            )
            metric.update(
                {
                    "seed": seed,
                    "schedule_kind": "single",
                    "intervention_t": intervention_t,
                    "recheck_t": math.nan,
                    **diagnostics,
                }
            )
            rows.append(metric)

        for intervention_t, recheck_t in paired_times:
            syn, _, runtime, diagnostics, _ = sample_selective(
                bundle,
                context,
                args.samples,
                args.steps,
                seed,
                "late_recheck_flagged",
                intervention_t,
                recheck_t,
                collect_forecast=False,
            )
            label = f"targeted_t{intervention_t:.2f}_t{recheck_t:.2f}"
            metric = core.evaluate_variant(
                label,
                syn,
                real,
                test,
                context,
                bundle,
                runtime,
                args.steps,
                diagnostics["correction_relation_events"],
            )
            metric.update(
                {
                    "seed": seed,
                    "schedule_kind": "targeted_recheck",
                    "intervention_t": intervention_t,
                    "recheck_t": recheck_t,
                    **diagnostics,
                }
            )
            rows.append(metric)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "timing_sweep_by_seed.csv", index=False)
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns if column != "seed"]
    summary = frame.groupby("variant")[numeric].agg(["mean", "std"])
    summary.columns = [f"{left}_{right}" for left, right in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "timing_sweep_summary.csv", index=False)
    metadata = {
        "status": "executed",
        "scope": "paired-seed timing sweep; oracle/rule critic, no LLM calls",
        "single_times": single_times,
        "paired_times": paired_times,
        "seeds": args.seeds,
        "samples_per_seed": args.samples,
        "steps": args.steps,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    columns = [
        "variant",
        "hard_any_violation_rate_mean",
        "numeric_ks_mean_mean",
        "categorical_tvd_mean_mean",
        "rare_pair_coverage_mean",
        "critic_record_evaluations_proxy_mean",
        "residual_among_initial_actions_mean",
    ]
    print(summary[columns].sort_values("hard_any_violation_rate_mean").to_string(index=False))


if __name__ == "__main__":
    main()
