#!/usr/bin/env python3
"""Aggregate conditional-adapter jobs and apply predeclared kill criteria."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
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
    for path in sorted(args.run_root.glob("condition_adapter_*")):
        if not path.is_dir() or not any(path.name.endswith(f"_{job_id}") for job_id in wanted):
            continue
        metrics = path / "contract_metrics.csv"
        meta = path / "run_metadata.json"
        item = {}
        if meta.exists():
            item = json.loads(meta.read_text()); item["run_dir"] = str(path); metadata.append(item)
        if metrics.exists():
            frame = pd.read_csv(metrics)
            frame["run_dir"] = str(path)
            frame["seed"] = item.get("seed")
            frames.append(frame)
    if not frames:
        raise RuntimeError("no completed adapter metrics found")
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(args.output / "all_contract_metrics.csv", index=False)
    summary = (
        all_metrics.groupby(["representation", "guidance_scale", "split"])
        .agg(
            contracts=("contract_id", "size"), satisfaction=("satisfaction_rate", "mean"),
            gain_frozen=("satisfaction_gain", "mean"), gain_null=("gain_over_finetuned_null", "mean"),
            cohort_ks=("cohort_numeric_ks_mean", "mean"), categorical_tv=("cohort_categorical_tv_mean", "mean"),
            coverage=("cohort_category_coverage_mean", "mean"), duplicate_rate=("duplicate_rate_rounded", "mean"),
        ).reset_index()
    )
    summary.to_csv(args.output / "comparison.csv", index=False)
    by_seed = (
        all_metrics.groupby(["representation", "guidance_scale", "seed", "split"])
        .agg(
            contracts=("contract_id", "size"), satisfaction=("satisfaction_rate", "mean"),
            gain_frozen=("satisfaction_gain", "mean"), gain_null=("gain_over_finetuned_null", "mean"),
            cohort_ks=("cohort_numeric_ks_mean", "mean"), categorical_tv=("cohort_categorical_tv_mean", "mean"),
            coverage=("cohort_category_coverage_mean", "mean"), duplicate_rate=("duplicate_rate_rounded", "mean"),
        ).reset_index()
    )
    by_seed.to_csv(args.output / "comparison_by_seed.csv", index=False)
    stability = (
        by_seed.groupby(["representation", "guidance_scale", "split"])
        .agg(
            seeds=("seed", "nunique"), gain_null_mean=("gain_null", "mean"), gain_null_std=("gain_null", "std"),
            satisfaction_mean=("satisfaction", "mean"), satisfaction_std=("satisfaction", "std"),
            cohort_ks_mean=("cohort_ks", "mean"), coverage_mean=("coverage", "mean"),
        ).reset_index()
    )
    stability.to_csv(args.output / "stability.csv", index=False)
    decisions = []
    for (representation, guidance), group in summary.groupby(["representation", "guidance_scale"]):
        threshold = group[group.split.eq("heldout_threshold")]
        composition = group[group.split.eq("heldout_composition")]
        threshold_gain = float(threshold.gain_null.iloc[0]) if len(threshold) else float("nan")
        composition_gain = float(composition.gain_null.iloc[0]) if len(composition) else float("nan")
        decisions.append(
            {
                "representation": representation,
                "guidance_scale": guidance,
                "heldout_threshold_gain_over_null": threshold_gain,
                "heldout_composition_gain_over_null": composition_gain,
                "passes_transfer_signal": bool(threshold_gain >= 0.05 and composition_gain >= 0.05),
            }
        )
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(args.output / "kill_criteria.csv", index=False)
    lines = [
        "# Conditional TabDiff adapter pilot", "",
        "Executed feasibility runs using the same pretrained checkpoint and training pairs.", "",
        "## Split-level comparison", "", markdown_table(summary), "",
        "## Transfer kill criterion", "", markdown_table(decision_frame), "",
        "## Seed stability", "", markdown_table(stability), "",
        "A representation passes the weak mechanism gate only if it improves both held-out-threshold and unseen-composition satisfaction by at least 0.05 over its own fine-tuned null condition. This is not yet a paper-level go decision; fidelity and seed replication remain required.", "",
        "Guardrail: request-ID has an all-zero vector for unseen requests, so it is a non-transfer baseline rather than a fair semantic encoder.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({"status": "executed", "requested_jobs": sorted(wanted), "loaded_runs": len(metadata), "runs": metadata}, indent=2))
    print(json.dumps({"status": "executed", "loaded_runs": len(metadata), "rows": len(all_metrics)}, indent=2))


if __name__ == "__main__":
    main()
