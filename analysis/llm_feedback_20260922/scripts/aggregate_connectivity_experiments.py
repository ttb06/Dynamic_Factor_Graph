#!/usr/bin/env python3
"""Aggregate paired graph-connectivity guidance experiments."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "macro_violation", "heldout_violation_error", "calibration_violation_error",
    "numeric_ks", "categorical_tv", "numeric_drift_sd", "affected_numeric_drift_sd",
    "action_events", "unique_rows_actuated", "runtime_seconds",
]


def relative_gain(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference if reference else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--variant-job", required=True)
    parser.add_argument("--trajectory-job", required=True)
    parser.add_argument("--guidance-job", required=True)
    parser.add_argument("--prefix", default="connectivity_guidance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    pattern = re.compile(re.escape(args.prefix) + r"_(.+)_gw([0-9.]+)_" + re.escape(args.guidance_job) + r"_\d+$")
    rows = []
    for directory in sorted((args.analysis / "outputs").glob(f"{args.prefix}_*_{args.guidance_job}_*")):
        match = pattern.fullmatch(directory.name)
        if not match or not (directory / "method_metrics.csv").exists():
            continue
        variant, weight = match.group(1), float(match.group(2))
        frame = pd.read_csv(directory / "method_metrics.csv")
        selected = frame[frame.method.eq("x0_reward_calibrated")]
        if selected.empty:
            continue
        row = selected.iloc[0].to_dict()
        rows.append({"variant": variant, "graph_weight": weight, "output": str(directory),
                     **{metric: row.get(metric, np.nan) for metric in METRICS}})
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("no completed connectivity guidance results")

    graph_summary = pd.read_csv(
        args.analysis / "outputs" / f"connectivity_variants_{args.variant_job}" / "graph_summary.csv"
    )
    results = results.merge(graph_summary, on="variant", how="left")
    results["macro_gain_vs_base_pct"] = np.nan
    results["heldout_gain_vs_base_pct"] = np.nan
    results["calibration_gain_vs_base_pct"] = np.nan
    for weight, block in results.groupby("graph_weight"):
        base = block[block.variant.eq("base")]
        if base.empty:
            continue
        base = base.iloc[0]
        index = block.index
        results.loc[index, "macro_gain_vs_base_pct"] = [
            relative_gain(float(base.macro_violation), float(value))
            for value in results.loc[index, "macro_violation"]
        ]
        results.loc[index, "heldout_gain_vs_base_pct"] = [
            relative_gain(float(base.heldout_violation_error), float(value))
            for value in results.loc[index, "heldout_violation_error"]
        ]
        results.loc[index, "calibration_gain_vs_base_pct"] = [
            relative_gain(float(base.calibration_violation_error), float(value))
            for value in results.loc[index, "calibration_violation_error"]
        ]
    results.sort_values(["graph_weight", "macro_violation", "heldout_violation_error"]).to_csv(
        args.output / "connectivity_results.csv", index=False
    )

    lines = [
        "# Graph size and connectivity guidance results", "",
        "All variants use the same checkpoint, sampling seed and fixed independent evaluation graph. "
        "Positive gain means lower error than the original 27-edge graph at the same graph weight.", "",
        "| Variant | Weight | Edges | Components | Largest | Hyperedges | Macro violation | Heldout error | Calibration error | Macro gain | Heldout gain | Num. KS | Drift | Actions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.sort_values(["graph_weight", "macro_violation"]).itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.graph_weight:g} | {int(row.edges)} | {int(row.components)} | "
            f"{int(row.largest_component)} | {int(row.multi_column_arity_ge_3)} | {row.macro_violation:.5f} | "
            f"{row.heldout_violation_error:.5f} | {row.calibration_violation_error:.5f} | "
            f"{row.macro_gain_vs_base_pct:+.1f}% | {row.heldout_gain_vs_base_pct:+.1f}% | "
            f"{row.numeric_ks:.5f} | {row.numeric_drift_sd:.5f} | {int(row.action_events)} |"
        )
    lines += ["", "Interpretation rule: a denser graph is useful only if semantic error falls without a material increase in distribution drift. "
              "If role_expanded helps but bridge_guardrails does not, the cause is guardrail coverage rather than connectivity. "
              "If dense_hypergraph wins over both, multi-column coordination is the stronger hypothesis."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "variant_job": args.variant_job,
        "trajectory_job": args.trajectory_job,
        "guidance_job": args.guidance_job,
        "fixed_evaluator": True,
        "paired_sampling_seed": True,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
