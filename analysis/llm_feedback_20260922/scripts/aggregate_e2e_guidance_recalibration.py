#!/usr/bin/env python3
"""Aggregate paired guidance-strength/action-budget calibration runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["pretrained"] * 6 + ["graph5"] * 6
STRENGTHS = ["weak", "weak", "moderate", "moderate", "base", "base"] * 2
BUDGETS = [1, 4, 1, 4, 1, 4] * 2


def gain(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference if reference else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--evaluation-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = args.analysis / "outputs"
    rows: list[dict] = []
    for index, (model, strength, budget) in enumerate(zip(MODELS, STRENGTHS, BUDGETS)):
        directory = outputs / f"e2e_guidance_recal_{model}_{strength}_a{budget}_{args.evaluation_job}_{index}"
        metrics = pd.read_csv(directory / "method_metrics.csv")
        metadata = pd.read_json(directory / "run_metadata.json", typ="series")
        for method in ("unconditional", "x0_reward_calibrated", "final_reward_calibrated"):
            selected = metrics.loc[metrics.method.eq(method)]
            if selected.empty:
                continue
            rows.append({
                "model": model, "strength": strength, "action_budget": budget, "method": method,
                "numeric_guidance_scale": float(metadata["numeric_guidance_scale"]),
                "max_guidance_scale": float(metadata["max_guidance_scale"]),
                **selected.iloc[0].to_dict(),
            })
    result = pd.DataFrame(rows)
    for metric in ("macro_violation", "heldout_violation_error", "calibration_violation_error"):
        result[f"{metric}_gain_vs_model_unconditional_pct"] = np.nan
        for model, block in result.groupby("model"):
            ref = float(block.loc[block.method.eq("unconditional"), metric].iloc[0])
            result.loc[block.index, f"{metric}_gain_vs_model_unconditional_pct"] = [gain(ref, float(value)) for value in block[metric]]
    result.to_csv(args.output / "guidance_recalibration.csv", index=False)

    primary = result.loc[result.method.eq("x0_reward_calibrated")].copy()
    lines = [
        "# Guidance recalibration after graph fine-tuning", "",
        "The same generation seed is used throughout. Each final-only row remains available in the CSV as a direct repair baseline.", "",
        "| Model | Strength | Max actions/row-edge | Macro | Heldout | Calibration | Num. KS | Cat. TV | Actions | Heldout gain vs own unconditional |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.strength} | {row.action_budget} | {row.macro_violation:.5f} | "
            f"{row.heldout_violation_error:.5f} | {row.calibration_violation_error:.5f} | {row.numeric_ks:.5f} | "
            f"{row.categorical_tv:.5f} | {int(row.action_events)} | {row.heldout_violation_error_gain_vs_model_unconditional_pct:+.1f}% |"
        )
    lines += ["", "A useful interaction requires graph fine-tuning to reach the same semantic error with fewer actions or lower strength, while preserving KS/TV."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
