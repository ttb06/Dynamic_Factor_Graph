#!/usr/bin/env python3
"""Aggregate calibrated active-set experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(args.input_root.glob("calibrated_gate_g*_t*_seed*/active_set_metrics.csv"))
    if not paths:
        raise FileNotFoundError("no calibrated active-set metrics")
    data = pd.concat(
        [pd.read_csv(path).assign(source=str(path.parent)) for path in paths], ignore_index=True
    )
    data.to_csv(args.output / "all_metrics.csv", index=False)
    summary = data.groupby(
        ["guidance_scale", "gate_start_t", "schedule", "early_policy", "gate_power", "split"],
        dropna=False,
    ).agg(
        seeds=("seed", "nunique"),
        violation=("conditional_violation_rate", "mean"),
        violation_std=("conditional_violation_rate", "std"),
        reduction=("conditional_violation_reduction", "mean"),
        antecedent_error=("antecedent_rate_error", "mean"),
        exception_error=("exception_rate_error", "mean"),
        inactive_error=("inactive_target_mean_error", "mean"),
        numeric_ks=("global_numeric_ks", "mean"),
        categorical_tv=("global_categorical_tv", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    lines = [
        "# Calibrated active-set results", "",
        "All deltas are target-scoped and all comparisons reuse paired initial noise.", "",
        "The train prior is population-level only. The paired-preview method runs an unconditional "
        "trajectory first and selectively reruns from the same initial noise.", "",
        "| CFG | gate start | method | split | seeds | violation | reduction | antecedent error | exception error | inactive error | numeric KS | categorical TV |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.guidance_scale:g} | {row.gate_start_t:g} | {row.schedule} | {row.split} | "
            f"{row.seeds} | {row.violation:.4f} | {row.reduction:.4f} | {row.antecedent_error:.4f} | "
            f"{row.exception_error:.4f} | {row.inactive_error:.4f} | {row.numeric_ks:.4f} | {row.categorical_tv:.4f} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"loaded {len(paths)} runs -> {args.output}")


if __name__ == "__main__":
    main()
