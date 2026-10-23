#!/usr/bin/env python3
"""Select guidance strength on one tuning seed using a predeclared objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    for path in sorted(args.outputs.glob(f"guidance_strength_*_{args.job_id}_*/method_metrics.csv")):
        run = path.parent.name
        prefix = "guidance_strength_"
        stem = run[len(prefix):].rsplit(f"_{args.job_id}_", 1)[0]
        dataset, variant = stem.split("_", 1)
        frame = pd.read_csv(path)
        guided = frame.loc[frame.method == "x0_reward_calibrated"].iloc[0]
        base = frame.loc[frame.method == "unconditional"].iloc[0]
        # The primary objective is calibration-target violation error.  Only
        # *additional* marginal distortion versus paired unconditional output
        # is penalized, so naturally difficult datasets are not punished.
        fidelity_excess = max(0.0, float(guided.numeric_ks-base.numeric_ks))
        fidelity_excess += max(0.0, float(guided.categorical_tv-base.categorical_tv))
        objective = float(guided.actuate_calibration_violation_error) + .25*fidelity_excess
        metadata = json.loads((path.parent/"run_metadata.json").read_text())
        rows.append({
            "dataset": dataset, "variant": variant, "objective": objective,
            "actuate_macro_violation": float(guided.actuate_macro_violation),
            "actuate_calibration_violation_error": float(guided.actuate_calibration_violation_error),
            "macro_violation": float(guided.macro_violation),
            "numeric_ks": float(guided.numeric_ks), "categorical_tv": float(guided.categorical_tv),
            "action_events": int(guided.action_events),
            "numeric_guidance_mode": metadata["numeric_guidance_mode"],
            "numeric_guidance_scale": metadata["numeric_guidance_scale"],
            "target_step_fraction": metadata["target_step_fraction"],
            "max_guidance_scale": metadata["max_guidance_scale"],
            "path": str(path.parent),
        })
    results = pd.DataFrame(rows)
    if len(results) != 21:
        raise RuntimeError(f"expected 21 completed tuning runs, found {len(results)}")
    results.to_csv(args.output/"all_tuning_results.csv", index=False)
    chosen = {}
    for dataset, group in results.groupby("dataset", sort=True):
        best = group.sort_values(["objective", "actuate_macro_violation", "variant"]).iloc[0]
        chosen[dataset] = {
            "variant": best.variant,
            "numeric_guidance_mode": best.numeric_guidance_mode,
            "numeric_guidance_scale": float(best.numeric_guidance_scale),
            "target_step_fraction": float(best.target_step_fraction),
            "max_guidance_scale": float(best.max_guidance_scale),
            "tuning_objective": float(best.objective),
        }
    (args.output/"selection.json").write_text(json.dumps(chosen, indent=2))
    lines = ["# Guidance-strength tuning", "",
             "One seed was used for selection. Confirmation seeds are separate. Objective = actuation calibration error + 0.25 × excess marginal distortion.", "",
             markdown_table(results.sort_values(["dataset", "objective"])), "",
             f"Selection: `{chosen}`"]
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
