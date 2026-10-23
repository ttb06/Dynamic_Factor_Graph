#!/usr/bin/env python3
"""Aggregate same-node paired adaptive-refresh ablation and verify base parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "actuate_heldout_violation_error", "heldout_violation_error", "numeric_ks",
    "categorical_tv", "numeric_drift_sd", "action_events", "runtime_seconds",
]
METHODS = ["dynamic", "hysteresis", "adaptive_hysteresis"]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def markdown(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(map(str, values.columns)) + " |",
        "| " + " | ".join(["---"] * len(values.columns)) + " |",
        *rows,
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.analysis / "outputs"
    rows, parity_rows = [], []
    for dataset in ["shoppers", "default", "diabetes"]:
        for seed in ["20261051", "20261052", "20261053"]:
            directories = {
                method: next(root.glob(f"adaptive_paired_{dataset}_{method}_s{seed}_{args.job}_*"))
                for method in METHODS
            }
            hashes = {method: digest(directory / "samples" / "unconditional.csv") for method, directory in directories.items()}
            parity_rows.append({"dataset": dataset, "seed": seed, **hashes, "all_equal": len(set(hashes.values())) == 1})
            for method, directory in directories.items():
                frame = pd.read_csv(directory / "method_metrics.csv")
                record = frame.loc[frame.method.eq("x0_reward_calibrated")].iloc[0].to_dict()
                trace = json.loads((directory / "trace_x0_reward_calibrated.json").read_text())
                counts = trace.get("adaptive_persistence_counts", {})
                count_total = sum(int(value) for value in counts.values())
                record["adaptive_p3_share"] = int(counts.get("3", 0)) / count_total if count_total else float("nan")
                metadata = json.loads((directory / "run_metadata.json").read_text())
                denominator = int(metadata["samples"]) * int(trace.get("topology_refreshes", 0))
                record["normalized_topology_flip_rate"] = int(trace.get("topology_flips", 0)) / denominator if denominator else float("nan")
                rows.append({**record, "dataset": dataset, "seed": seed, "method_variant": method})

    per_seed = pd.DataFrame(rows)
    parity = pd.DataFrame(parity_rows)
    per_seed.to_csv(args.output / "paired_per_seed.csv", index=False)
    parity.to_csv(args.output / "base_parity.csv", index=False)
    summary = per_seed.groupby(["dataset", "method_variant"])[
        METRICS + ["adaptive_p3_share", "normalized_topology_flip_rate"]
    ].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(value).rstrip("_") if isinstance(value, tuple) else value for value in summary.columns]
    summary.to_csv(args.output / "paired_summary.csv", index=False)
    columns = [
        "dataset", "method_variant", "actuate_heldout_violation_error_mean",
        "actuate_heldout_violation_error_std", "heldout_violation_error_mean",
        "numeric_ks_mean", "categorical_tv_mean", "action_events_mean",
        "adaptive_p3_share_mean", "normalized_topology_flip_rate_mean", "runtime_seconds_mean",
    ]
    lines = [
        "# Same-node paired volatility-adaptive ablation", "",
        f"Unconditional sample parity passed: {int(parity.all_equal.sum())}/{len(parity)} dataset-seed groups.", "",
        markdown(summary[columns]), "",
    ]
    for dataset in ["shoppers", "default", "diabetes"]:
        view = per_seed.loc[per_seed.dataset.eq(dataset)].pivot(index="seed", columns="method_variant", values="actuate_heldout_violation_error")
        lines.append(
            f"- {dataset}: adaptive beats dynamic in {int((view.adaptive_hysteresis < view.dynamic).sum())}/3 and fixed hysteresis-3 in {int((view.adaptive_hysteresis < view.hysteresis).sum())}/3 seeds."
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
