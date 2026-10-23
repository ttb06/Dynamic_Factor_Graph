#!/usr/bin/env python3
"""Aggregate the graph-finetuning checkpoint frontier on a fixed evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPOCHS = [100, 200, 300, 400, 500]
METHODS = ["unconditional", "x0_reward_calibrated", "final_reward_calibrated"]


def gain(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference if reference else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--trajectory-job", required=True)
    parser.add_argument("--evaluation-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = args.analysis / "outputs"

    baseline_path = outputs / "e2e_graph_eval_pretrained_1062232_0" / "method_metrics.csv"
    baseline = pd.read_csv(baseline_path)
    rows: list[dict] = []
    for method in METHODS:
        row = baseline.loc[baseline.method.eq(method)]
        if row.empty:
            continue
        rows.append({"epoch": 0, "method": method, "mean_readability_cutoff": 0.291,
                     **row.iloc[0].to_dict()})

    for index, epoch in enumerate(EPOCHS):
        metrics = pd.read_csv(outputs / f"e2e_checkpoint_eval_e{epoch}_{args.evaluation_job}_{index}" / "method_metrics.csv")
        schedule = pd.read_csv(outputs / f"e2e_checkpoint_trajectory_e{epoch}_{args.trajectory_job}_{index}" / "typed_readability_schedule.csv")
        cutoff = float(schedule.loc[schedule.readable.astype(bool), "cutoff"].mean())
        for method in METHODS:
            row = metrics.loc[metrics.method.eq(method)]
            if row.empty:
                continue
            rows.append({"epoch": epoch, "method": method, "mean_readability_cutoff": cutoff,
                         **row.iloc[0].to_dict()})

    result = pd.DataFrame(rows)
    for metric in ("macro_violation", "heldout_violation_error", "calibration_violation_error", "numeric_ks"):
        result[f"{metric}_gain_vs_pretrained_pct"] = np.nan
        for method, block in result.groupby("method"):
            ref = block.loc[block.epoch.eq(0), metric]
            if ref.empty:
                continue
            result.loc[block.index, f"{metric}_gain_vs_pretrained_pct"] = [gain(float(ref.iloc[0]), float(value)) for value in block[metric]]
    result.to_csv(args.output / "checkpoint_frontier.csv", index=False)

    lines = [
        "# Graph fine-tuning checkpoint frontier", "",
        "Epoch 0 is the original pretrained TabDiff checkpoint. All rows use the same generation seed and independent evaluation graph.", "",
        "| Epoch | Method | Macro | Heldout | Calibration | Num. KS | Cat. TV | Readable t | Actions | Heldout gain | KS change |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.epoch} | {row.method} | {row.macro_violation:.5f} | {row.heldout_violation_error:.5f} | "
            f"{row.calibration_violation_error:.5f} | {row.numeric_ks:.5f} | {row.categorical_tv:.5f} | "
            f"{row.mean_readability_cutoff:.3f} | {int(row.action_events)} | "
            f"{row.heldout_violation_error_gain_vs_pretrained_pct:+.1f}% | {-row.numeric_ks_gain_vs_pretrained_pct:+.1f}% |"
        )
    lines += ["", "Selection must use calibration/fidelity metrics, not the held-out audit metric. The held-out column is reported only for final evaluation."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
