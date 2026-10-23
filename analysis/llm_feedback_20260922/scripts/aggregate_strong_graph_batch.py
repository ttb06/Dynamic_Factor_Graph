#!/usr/bin/env python3
"""Aggregate the strong-topology controls and dynamic-plus-repair runs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = [
    "macro_violation",
    "heldout_violation_error",
    "calibration_violation_error",
    "numeric_ks",
    "categorical_tv",
    "numeric_drift_sd",
    "action_events",
    "unique_rows_actuated",
]


def metric_row(path: Path, method: str) -> dict:
    frame = pd.read_csv(path / "method_metrics.csv")
    match = frame.loc[frame["method"].eq(method)]
    if len(match) != 1:
        raise RuntimeError(f"expected one {method} row in {path}, found {len(match)}")
    row = match.iloc[0]
    return {key: row[key] for key in METRICS}


def markdown(frame: pd.DataFrame) -> str:
    def value(item):
        if isinstance(item, float):
            return f"{item:.6f}"
        return str(item)

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(value(item) for item in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--control-job", required=True)
    parser.add_argument("--combined-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    controls = []
    pattern = re.compile(r"strong_full_control_(\w+)_(dynamic|frozen|frozen_state|shuffled|oracle_final)_s(\d+)_")
    for path in sorted(args.outputs.glob(f"strong_full_control_*_{args.control_job}_*")):
        match = pattern.match(path.name)
        if not match:
            continue
        dataset, topology, seed = match.groups()
        controls.append({"dataset": dataset, "seed": int(seed), "topology": topology,
                         **metric_row(path, "x0_reward_calibrated")})
    control_frame = pd.DataFrame(controls)
    if control_frame.empty:
        raise RuntimeError("no topology-control outputs found")
    control_frame.to_csv(args.output / "topology_controls.csv", index=False)

    combined = []
    pattern = re.compile(r"combined_confirm_(\w+)_(c[0-9.]+)_s(\d+)_")
    for path in sorted(args.outputs.glob(f"combined_confirm_*_{args.combined_job}_*")):
        match = pattern.match(path.name)
        if not match:
            continue
        dataset, variant, seed = match.groups()
        online = metric_row(path, "x0_reward_calibrated")
        for method in ["unconditional", "x0_reward_calibrated", "x0_reward_calibrated_then_final_reward", "final_reward_calibrated"]:
            values = metric_row(path, method)
            combined.append({"dataset": dataset, "seed": int(seed), "variant": variant,
                             "method": method, **values,
                             "residual_repair_events": (values["action_events"] - online["action_events"]
                                                        if method == "x0_reward_calibrated_then_final_reward" else 0)})
    combined_frame = pd.DataFrame(combined)
    if combined_frame.empty:
        raise RuntimeError("no combined-confirmation outputs found")
    combined_frame.to_csv(args.output / "combined_confirmation.csv", index=False)

    control_mean = control_frame.groupby(["dataset", "topology"], as_index=False)[METRICS].mean()
    combined_mean = combined_frame.groupby(["dataset", "variant", "method"], as_index=False)[
        METRICS + ["residual_repair_events"]
    ].mean()
    lines = [
        "# Strong graph batch",
        "",
        f"Topology job: `{args.control_job}`; combined-repair job: `{args.combined_job}`.",
        "",
        "## Topology controls (mean across available seeds)",
        "",
        markdown(control_mean),
        "",
        "## Dynamic guidance and residual repair (mean across available seeds)",
        "",
        markdown(combined_mean),
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
