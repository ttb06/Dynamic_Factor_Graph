#!/usr/bin/env python3
"""Aggregate locked-scope dynamic-graph actuator variants."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from evaluate_conditional_topology_controls import audit_sample


PATTERN = re.compile(
    r"conditional_actuator_(?P<request>range|category|both)_"
    r"m(?P<margin>[^_]+)_s(?P<scale>[^_]+)_h(?P<hold>[^_]+)_"
    r"(?P<job>\d+)_(?P<task>\d+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for directory in sorted(args.root.glob(f"conditional_actuator_*_{args.job_id}_*")):
        match = PATTERN.fullmatch(directory.name)
        if not match:
            continue
        request = match.group("request")
        base = pd.read_csv(directory / "samples/unconditional.csv")
        sample = pd.read_csv(directory / "samples/xstate_reward_calibrated.csv")
        metric = pd.read_csv(directory / "method_metrics.csv").set_index("method").loc["xstate_reward_calibrated"]
        rows.append({
            "request": request,
            "margin_sd": float(match.group("margin")),
            "guidance_scale": float(match.group("scale")),
            "hold_steps": int(match.group("hold")),
            **audit_sample(base, sample, request),
            **{key: float(metric[key]) for key in [
                "numeric_ks", "categorical_tv", "numeric_drift_sd",
                "action_events", "unique_rows_actuated", "runtime_seconds",
            ]},
            "directory": str(directory),
        })
    result = pd.DataFrame(rows).sort_values(["request", "violation_given_applicable", "numeric_ks"])
    result.to_csv(args.output / "actuator_sweep.csv", index=False)
    lines = [
        "# Locked-scope actuator sweep", "",
        "| Request | Margin SD | Guidance | Hold | Violation in scope | Scope loss | Outside-target edits | Numeric KS | Categorical TV | Actions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.request} | {row.margin_sd:g} | {row.guidance_scale:g} | {row.hold_steps} | "
            f"{row.violation_given_applicable:.2%} | {row.antecedent_evasion_given_base_applicable:.2%} | "
            f"{row.target_changed_given_base_nonapplicable:.2%} | {row.numeric_ks:.4f} | "
            f"{row.categorical_tv:.4f} | {int(row.action_events)} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
