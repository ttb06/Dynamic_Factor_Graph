#!/usr/bin/env python3
"""Aggregate same-node end-to-end transition-certifier policies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


POLICIES = ["immediate", "wait3", "dynamic_wait5", "topology_wait5", "frozen_wait5"]
METRICS = [
    "actuate_heldout_violation_error", "heldout_violation_error", "numeric_ks",
    "categorical_tv", "numeric_drift_sd", "action_events", "runtime_seconds",
]


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
        for seed in ["20261071", "20261072", "20261073"]:
            directories = {
                policy: next(root.glob(f"transition_policy_{dataset}_{policy}_s{seed}_{args.job}_*"))
                for policy in POLICIES
            }
            hashes = {
                policy: digest(directory / "samples" / "unconditional.csv")
                for policy, directory in directories.items()
            }
            parity_rows.append({
                "dataset": dataset,
                "seed": seed,
                **hashes,
                "all_equal": len(set(hashes.values())) == 1,
            })
            for policy, directory in directories.items():
                frame = pd.read_csv(directory / "method_metrics.csv")
                record = frame.loc[frame.method.eq("x0_reward_calibrated")].iloc[0].to_dict()
                trace = json.loads((directory / "trace_x0_reward_calibrated.json").read_text())
                record["topology_flips"] = int(trace.get("topology_flips", 0))
                record["topology_refreshes"] = int(trace.get("topology_refreshes", 0))
                rows.append({**record, "dataset": dataset, "seed": seed, "policy": policy})

    per_seed = pd.DataFrame(rows)
    parity = pd.DataFrame(parity_rows)
    if not bool(parity.all_equal.all()):
        raise RuntimeError("unconditional sample parity failed")
    per_seed.to_csv(args.output / "paired_per_seed.csv", index=False)
    parity.to_csv(args.output / "base_parity.csv", index=False)
    summary = per_seed.groupby(["dataset", "policy"])[METRICS + ["topology_flips", "topology_refreshes"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(value).rstrip("_") if isinstance(value, tuple) else value for value in summary.columns]
    summary.to_csv(args.output / "paired_summary.csv", index=False)

    selected_policy = {"shoppers": "dynamic_wait5", "default": "topology_wait5", "diabetes": "topology_wait5"}
    tests = []
    for dataset, certified in selected_policy.items():
        view = per_seed[per_seed.dataset.eq(dataset)].pivot(index="seed", columns="policy", values="actuate_heldout_violation_error")
        tests.append({
            "dataset": dataset,
            "certified_policy": certified,
            "certified_mean": float(view[certified].mean()),
            "immediate_mean": float(view.immediate.mean()),
            "wait3_mean": float(view.wait3.mean()),
            "frozen_mean": float(view.frozen_wait5.mean()),
            "wins_vs_immediate": int((view[certified] < view.immediate).sum()),
            "wins_vs_wait3": int((view[certified] < view.wait3).sum()),
            "wins_vs_frozen": int((view[certified] < view.frozen_wait5).sum()),
        })
    tests_frame = pd.DataFrame(tests)
    tests_frame.to_csv(args.output / "certified_policy_tests.csv", index=False)
    columns = [
        "dataset", "policy", "actuate_heldout_violation_error_mean",
        "actuate_heldout_violation_error_std", "heldout_violation_error_mean",
        "numeric_ks_mean", "categorical_tv_mean", "action_events_mean",
        "topology_flips_mean", "runtime_seconds_mean",
    ]
    lines = [
        "# Transition-certified online policy", "",
        f"Unconditional sample parity passed: {int(parity.all_equal.sum())}/{len(parity)} dataset-seed groups.", "",
        markdown(summary[columns]), "", "## Pre-registered certified policy comparison", "",
        markdown(tests_frame), "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
