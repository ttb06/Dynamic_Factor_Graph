#!/usr/bin/env python3
"""Aggregate conditional-implication adapter runs across modes and seeds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value): values.append("")
            elif isinstance(value, float): values.append(f"{value:.4f}")
            else: values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    wanted = {token for token in re.split(r"[:,]", args.job_ids) if token}
    frames, metadata = [], []
    for path in sorted(args.run_root.glob("implication_adapter_*")):
        if not any(path.name.endswith(f"_{job_id}") for job_id in wanted): continue
        meta_path, metric_path = path / "run_metadata.json", path / "contract_metrics.csv"
        if not meta_path.exists() or not metric_path.exists(): continue
        meta = json.loads(meta_path.read_text()); meta["run_dir"] = str(path); metadata.append(meta)
        frame = pd.read_csv(metric_path); frame["run_dir"] = str(path); frames.append(frame)
    if not frames: raise RuntimeError("no completed implication runs")
    metrics = pd.concat(frames, ignore_index=True); metrics.to_csv(args.output / "all_contract_metrics.csv", index=False)
    conditioned = metrics[metrics.method.eq("conditioned")].copy()
    by_seed = conditioned.groupby(["training_mode", "guidance_scale", "seed", "split"]).agg(
        contracts=("contract_id", "size"), conditional_violation=("conditional_violation_rate", "mean"),
        violation_reduction=("conditional_violation_reduction", "mean"), antecedent_error=("antecedent_rate_error", "mean"),
        retention=("antecedent_retention_ratio", "mean"), exception_error=("exception_rate_error", "mean"),
        inactive_target_error=("inactive_target_mean_error", "mean"), boundary_mass=("target_boundary_mass_active", "mean"),
        numeric_ks=("global_numeric_ks", "mean"), categorical_tv=("global_categorical_tv", "mean"),
    ).reset_index()
    by_seed.to_csv(args.output / "comparison_by_seed.csv", index=False)
    stability = by_seed.groupby(["training_mode", "guidance_scale", "split"]).agg(
        seeds=("seed", "nunique"), violation_mean=("conditional_violation", "mean"), violation_std=("conditional_violation", "std"),
        reduction_mean=("violation_reduction", "mean"), antecedent_error_mean=("antecedent_error", "mean"),
        antecedent_error_std=("antecedent_error", "std"), exception_error_mean=("exception_error", "mean"),
        inactive_error_mean=("inactive_target_error", "mean"), boundary_mass_mean=("boundary_mass", "mean"),
        numeric_ks_mean=("numeric_ks", "mean"), categorical_tv_mean=("categorical_tv", "mean"),
    ).reset_index()
    stability.to_csv(args.output / "stability.csv", index=False)
    pivot = by_seed.pivot_table(
        index=["guidance_scale", "seed", "split"], columns="training_mode",
        values=["conditional_violation", "antecedent_error", "exception_error", "inactive_target_error"],
    )
    paired_rows = []
    for index, row in pivot.iterrows():
        guidance, seed, split = index
        if ("antecedent_error", "filter") not in row or ("antecedent_error", "repair_preserve") not in row: continue
        paired_rows.append(
            {
                "guidance_scale": guidance, "seed": seed, "split": split,
                "antecedent_error_improvement": row[("antecedent_error", "filter")] - row[("antecedent_error", "repair_preserve")],
                "violation_change_repair_minus_filter": row[("conditional_violation", "repair_preserve")] - row[("conditional_violation", "filter")],
                "exception_error_improvement": row[("exception_error", "filter")] - row[("exception_error", "repair_preserve")],
                "inactive_error_improvement": row[("inactive_target_error", "filter")] - row[("inactive_target_error", "repair_preserve")],
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.output / "paired_mode_differences.csv", index=False)
    paired_summary = paired.groupby(["guidance_scale", "split"]).agg(
        seeds=("seed", "nunique"), antecedent_error_improvement=("antecedent_error_improvement", "mean"),
        violation_change=("violation_change_repair_minus_filter", "mean"),
        exception_error_improvement=("exception_error_improvement", "mean"),
        inactive_error_improvement=("inactive_error_improvement", "mean"),
    ).reset_index() if len(paired) else pd.DataFrame()
    paired_summary.to_csv(args.output / "paired_summary.csv", index=False)
    lines = [
        "# Conditional implication and antecedent-evasion pilot", "", "## Stability", "", table(stability), "",
        "## Repair-preserve minus filter trade-off", "", table(paired_summary) if len(paired_summary) else "No paired runs.", "",
        "Positive antecedent/exception/inactive improvements favor repair-preserve; negative violation change favors repair-preserve. "
        "The mechanism is useful only if violation falls without replacing the antecedent subgroup by not-applicable or exception rows.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({"status": "executed", "loaded_runs": len(metadata), "runs": metadata}, indent=2))
    print(json.dumps({"status": "executed", "loaded_runs": len(metadata), "metric_rows": len(metrics)}, indent=2))


if __name__ == "__main__": main()
