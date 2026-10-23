#!/usr/bin/env python3
"""Aggregate topology controls at the selected scale-2 T4 operating point."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODELS = ["pretrained", "epoch100"]
TOPOLOGIES = ["dynamic", "frozen", "frozen_state"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--evaluation-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    rows = []; index = 0
    for model in MODELS:
        for topology in TOPOLOGIES:
            directory = args.analysis / "outputs" / f"e2e_t4_topology_{model}_{topology}_{args.evaluation_job}_{index}"
            metrics = pd.read_csv(directory / "method_metrics.csv")
            row = metrics.loc[metrics.method.eq("x0_reward_calibrated")].iloc[0]
            rows.append({"model": model, "topology": topology, **row.to_dict()})
            index += 1
    result = pd.DataFrame(rows); result.to_csv(args.output / "topology_results.csv", index=False)
    lines = ["# T4 topology controls at scale 2", "",
             "| Model | Topology | Macro | Heldout | Calibration | Num. KS | Cat. TV | Actions |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result.itertuples(index=False):
        lines.append(f"| {row.model} | {row.topology} | {row.macro_violation:.5f} | {row.heldout_violation_error:.5f} | "
                     f"{row.calibration_violation_error:.5f} | {row.numeric_ks:.5f} | {row.categorical_tv:.5f} | {int(row.action_events)} |")
    lines += ["", "This is the direct contribution test: changing the active graph along the trajectory must outperform freezing it under an identical action mechanism and budget."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
