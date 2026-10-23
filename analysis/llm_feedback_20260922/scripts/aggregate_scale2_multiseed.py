#!/usr/bin/env python3
"""Three-seed confirmation for the selected scale-2 Shoppers controller."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SEEDS = [20261301, 20261311, 20261312]


def selected(path: Path, method: str) -> dict:
    frame = pd.read_csv(path / "method_metrics.csv")
    return frame.loc[frame.method.eq(method)].iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--evaluation-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    outputs = args.analysis / "outputs"; rows = []
    for seed_index, seed in enumerate(SEEDS):
        if seed_index == 0:
            dynamic = outputs / "e2e_t4_topology_pretrained_dynamic_1062383_0"
            frozen = outputs / "e2e_t4_topology_pretrained_frozen_1062383_1"
        else:
            offset = 2 * (seed_index - 1)
            dynamic = outputs / f"main_scale2_confirm_dynamic_s{seed}_{args.evaluation_job}_{offset}"
            frozen = outputs / f"main_scale2_confirm_frozen_s{seed}_{args.evaluation_job}_{offset + 1}"
        for name, path, method in (
            ("unconditional", dynamic, "unconditional"),
            ("final_repair", dynamic, "final_reward_calibrated"),
            ("dynamic", dynamic, "x0_reward_calibrated"),
            ("frozen", frozen, "x0_reward_calibrated"),
        ):
            rows.append({**selected(path, method), "seed": seed, "method": name})
    result = pd.DataFrame(rows); result.to_csv(args.output / "per_seed.csv", index=False)
    metrics = ["macro_violation", "heldout_violation_error", "calibration_violation_error",
               "numeric_ks", "categorical_tv", "action_events"]
    summary = result.groupby("method")[metrics].agg(["mean", "std"])
    summary.to_csv(args.output / "summary.csv")
    paired = []
    for seed in SEEDS:
        block = result.loc[result.seed.eq(seed)].set_index("method")
        paired.append({
            "seed": seed,
            "dynamic_macro_gain_vs_frozen_pct": 100 * (block.loc["frozen", "macro_violation"] - block.loc["dynamic", "macro_violation"]) / block.loc["frozen", "macro_violation"],
            "dynamic_heldout_gain_vs_frozen_pct": 100 * (block.loc["frozen", "heldout_violation_error"] - block.loc["dynamic", "heldout_violation_error"]) / block.loc["frozen", "heldout_violation_error"],
            "dynamic_heldout_gap_vs_final_pct": 100 * (block.loc["dynamic", "heldout_violation_error"] - block.loc["final_repair", "heldout_violation_error"]) / block.loc["final_repair", "heldout_violation_error"],
        })
    paired_frame = pd.DataFrame(paired); paired_frame.to_csv(args.output / "paired_gains.csv", index=False)
    lines = ["# Scale-2 Shoppers confirmation (3 seeds)", "",
             "| Method | Macro | Heldout | Calibration | Num. KS | Cat. TV | Actions |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in ["unconditional", "final_repair", "dynamic", "frozen"]:
        block = result.loc[result.method.eq(method)]
        values = [f"{block[column].mean():.5f} +/- {block[column].std():.5f}" for column in metrics[:-1]]
        lines.append(f"| {method} | " + " | ".join(values) + f" | {block.action_events.mean():.1f} |")
    lines += ["", f"Dynamic vs frozen heldout gain: {paired_frame.dynamic_heldout_gain_vs_frozen_pct.mean():.2f}% +/- {paired_frame.dynamic_heldout_gain_vs_frozen_pct.std():.2f}%.",
              f"Dynamic heldout gap versus final repair: {paired_frame.dynamic_heldout_gap_vs_final_pct.mean():+.2f}% +/- {paired_frame.dynamic_heldout_gap_vs_final_pct.std():.2f}%."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
