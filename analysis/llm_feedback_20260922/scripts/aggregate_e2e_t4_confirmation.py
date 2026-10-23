#!/usr/bin/env python3
"""Aggregate the architecture-matched end-to-end/controller confirmation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODELS = ["pretrained", "epoch100", "epoch500"]
PROFILES = [("s1_a4", 1.0, 4), ("s2_a4", 2.0, 4), ("s4_a1", 4.0, 1), ("s4_a4", 4.0, 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--evaluation-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    index = 0
    for model in MODELS:
        for profile, scale, budget in PROFILES:
            directory = args.analysis / "outputs" / f"e2e_t4_confirm_{model}_{profile}_{args.evaluation_job}_{index}"
            metrics = pd.read_csv(directory / "method_metrics.csv")
            for method in ("unconditional", "x0_reward_calibrated", "final_reward_calibrated"):
                selected = metrics.loc[metrics.method.eq(method)]
                if selected.empty:
                    continue
                rows.append({"model": model, "profile": profile, "scale": scale,
                             "action_budget": budget, "method": method, **selected.iloc[0].to_dict()})
            index += 1
    result = pd.DataFrame(rows)
    for metric in ("macro_violation", "heldout_violation_error", "calibration_violation_error"):
        result[f"{metric}_gain_vs_own_unconditional_pct"] = 0.0
        for (model, profile), block in result.groupby(["model", "profile"]):
            reference = float(block.loc[block.method.eq("unconditional"), metric].iloc[0])
            result.loc[block.index, f"{metric}_gain_vs_own_unconditional_pct"] = 100 * (reference - block[metric]) / reference
    result.to_csv(args.output / "t4_confirmation.csv", index=False)
    primary = result.loc[result.method.eq("x0_reward_calibrated")]
    lines = [
        "# T4-matched GraphFT/controller confirmation", "",
        "All tasks request the same T4 GPU architecture; every guided sample is paired with its own unconditional sample.", "",
        "| Model | Profile | Macro | Heldout | Calibration | Num. KS | Cat. TV | Actions | Heldout gain |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.profile} | {row.macro_violation:.5f} | {row.heldout_violation_error:.5f} | "
            f"{row.calibration_violation_error:.5f} | {row.numeric_ks:.5f} | {row.categorical_tv:.5f} | "
            f"{int(row.action_events)} | {row.heldout_violation_error_gain_vs_own_unconditional_pct:+.1f}% |"
        )
    lines += ["", "GraphFT is retained only if epoch100 or epoch500 improves the semantic/fidelity frontier or matches pretrained validity with materially fewer actions."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
