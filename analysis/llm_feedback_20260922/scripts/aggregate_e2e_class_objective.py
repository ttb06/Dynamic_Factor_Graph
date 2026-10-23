#!/usr/bin/env python3
"""Aggregate semantic-hardness variants of the graph fine-tuning loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VARIANTS = ["hard5", "hard10", "near5"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--train-job", required=True)
    parser.add_argument("--trajectory-job", required=True)
    parser.add_argument("--evaluation-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = args.analysis / "outputs"
    rows: list[dict] = []
    for index, variant in enumerate(VARIANTS):
        metrics = pd.read_csv(outputs / f"e2e_class_eval_{variant}_{args.evaluation_job}_{index}" / "method_metrics.csv")
        schedule = pd.read_csv(outputs / f"e2e_class_trajectory_{variant}_{args.trajectory_job}_{index}" / "typed_readability_schedule.csv")
        metadata = json.loads((outputs / f"e2e_class_finetune_{variant}_{args.train_job}_{index}" / "training_metadata.json").read_text())
        for method in ("unconditional", "x0_reward_calibrated", "final_reward_calibrated"):
            selected = metrics.loc[metrics.method.eq(method)]
            if selected.empty:
                continue
            rows.append({
                "variant": variant, "method": method,
                "graph_weight": metadata["graph_weight"],
                "classifications": metadata["allowed_classifications"],
                "compiled_edges": metadata["compiled_edges"],
                "parameter_displacement": metadata["relative_parameter_displacement"],
                "mean_readability_cutoff": float(schedule.loc[schedule.readable.astype(bool), "cutoff"].mean()),
                **selected.iloc[0].to_dict(),
            })
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "class_objective_results.csv", index=False)
    lines = [
        "# Hard versus near-hard graph objective", "",
        "All rows use the same initialization, middle timestep window, generation seed and independent evaluation graph.", "",
        "| Variant | Classes | Edges | Method | Macro | Heldout | Calibration | Num. KS | Cat. TV | Actions |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.classifications} | {row.compiled_edges} | {row.method} | "
            f"{row.macro_violation:.5f} | {row.heldout_violation_error:.5f} | {row.calibration_violation_error:.5f} | "
            f"{row.numeric_ks:.5f} | {row.categorical_tv:.5f} | {int(row.action_events)} |"
        )
    lines += ["", "This isolates whether uncertain near-hard relations, rather than end-to-end training itself, are responsible for held-out and fidelity degradation."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
