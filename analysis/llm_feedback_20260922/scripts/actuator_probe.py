#!/usr/bin/env python3
"""Compare mixed-type correction actuators under paired seeds."""

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


VARIANTS = (
    ("x0_single_t0.03", "late_once", 0.03, 0.03, "x0_prediction"),
    ("state_single_t0.03", "late_once", 0.03, 0.03, "state_projection"),
    ("x0_single_t0.10", "late_once", 0.10, 0.03, "x0_prediction"),
    ("state_single_t0.10", "late_once", 0.10, 0.03, "state_projection"),
    ("x0_recheck_t0.10_t0.03", "late_recheck_flagged", 0.10, 0.03, "x0_prediction"),
    ("state_recheck_t0.10_t0.03", "late_recheck_flagged", 0.10, 0.03, "state_projection"),
)


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

    rows = []
    for seed in args.seeds:
        no_guidance = None
        for label, mode, intervention_t, recheck_t, actuator in VARIANTS:
            syn, _, runtime, diagnostics, _ = sample_selective(
                bundle,
                context,
                args.samples,
                args.steps,
                seed,
                mode,
                intervention_t,
                recheck_t,
                collect_forecast=False,
                numeric_actuator=actuator,
            )
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
                    "numeric_actuator": actuator,
                    "intervention_t": intervention_t,
                    "recheck_t": recheck_t if mode == "late_recheck_flagged" else math.nan,
                    **diagnostics,
                }
            )
            rows.append(metric)

        no_guidance, _, runtime, diagnostics, _ = sample_selective(
            bundle,
            context,
            args.samples,
            args.steps,
            seed,
            "no_guidance",
            0.10,
            0.03,
            collect_forecast=False,
        )
        baseline = core.evaluate_variant(
            "no_guidance", no_guidance, real, test, context, bundle, runtime, args.steps, 0
        )
        baseline.update({"seed": seed, "numeric_actuator": "none", **diagnostics})
        rows.append(baseline)
        repaired = core.post_repair(no_guidance, context)
        repaired_metric = core.evaluate_variant(
            "post_repair", repaired, real, test, context, bundle, 0.0, args.steps, 0
        )
        repaired_metric.update(
            {
                "seed": seed,
                "numeric_actuator": "post_generation_projection",
                "critic_record_evaluations_proxy": 0,
                "residual_among_initial_actions": 0.0,
            }
        )
        rows.append(repaired_metric)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "actuator_metrics_by_seed.csv", index=False)
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns if column != "seed"]
    summary = frame.groupby("variant")[numeric].agg(["mean", "std"])
    summary.columns = [f"{left}_{right}" for left, right in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "actuator_metrics_summary.csv", index=False)
    metadata = {
        "status": "executed",
        "scope": "typed actuator mechanism probe; rule/oracle critic, no LLM calls",
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
        "violation_education_fd_mean",
        "numeric_ks_mean_mean",
        "categorical_tvd_mean_mean",
        "rare_pair_coverage_mean",
        "critic_record_evaluations_proxy_mean",
        "residual_education_fd_rate_mean",
    ]
    print(summary[columns].sort_values("hard_any_violation_rate_mean").to_string(index=False))


if __name__ == "__main__":
    main()
