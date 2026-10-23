#!/usr/bin/env python3
"""Aggregate the matched Harpoon-style natural-graph sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "macro_violation", "heldout_violation_error", "calibration_violation_error",
    "numeric_ks", "categorical_tv", "requested_guidance", "editable_guidance",
    "runtime_seconds",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--dynamic-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    for run in args.runs:
        metadata = json.loads((run / "run_metadata.json").read_text())
        frame = pd.read_csv(run / "method_metrics.csv")
        frame.insert(0, "seed", int(metadata["seed"]))
        frame.insert(1, "run", str(run))
        rows.append(frame)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(args.output / "per_seed_metrics.csv", index=False)

    summary_rows = []
    for method, group in raw.groupby("method", sort=False):
        row = {"method": method, "seeds": group.seed.nunique()}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    selected = {}
    for policy in ("hard", "calibrated"):
        candidates = summary[summary.method.str.startswith(f"harpoon_{policy}_w") &
                             ~summary.method.str.endswith("w0")]
        # The graph's relations are probabilistic. Select only by the held-in
        # calibration partition, never by held-out semantic error.
        best = candidates.sort_values(["calibration_violation_error_mean", "categorical_tv_mean"]).iloc[0]
        selected[policy] = {
            "method": str(best.method),
            "selection_metric": "mean calibration_violation_error",
            "selection_uses_heldout": False,
        }
    (args.output / "selected_configs.json").write_text(json.dumps(selected, indent=2))

    dynamic = pd.read_csv(args.dynamic_summary)
    references = []
    unconditional = summary.loc[summary.method.eq("unconditional")].iloc[0]
    references.append({"method": "TabDiff unconditional",
                       **{metric: float(unconditional[f"{metric}_mean"])
                          for metric in METRICS[:5]}})
    dynamic_row = dynamic.loc[dynamic.variant.eq("dynamic")].iloc[0]
    references.append({"method": "Dynamic graph",
                       **{metric: float(dynamic_row[f"{metric}_mean"])
                          for metric in METRICS[:5]}})
    for policy, item in selected.items():
        row = summary.loc[summary.method.eq(item["method"])].iloc[0]
        references.append({"method": f"Harpoon-style {policy}",
                           **{metric: float(row[f"{metric}_mean"])
                              for metric in METRICS[:5]}})
    comparison = pd.DataFrame(references)
    comparison.to_csv(args.output / "selected_comparison.csv", index=False)

    lines = ["# Matched Harpoon-style natural-graph comparison", "",
             "The Harpoon rows are local TabDiff ports, not official Harpoon reproductions. "
             "Strength is selected by train-only calibration error averaged across sampling seeds.", "",
             "| Method | Macro violation | Held-out error | Calibration error | Numeric KS | Categorical TV |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison.itertuples(index=False):
        lines.append(f"| {row.method} | {row.macro_violation:.4f} | {row.heldout_violation_error:.4f} | "
                     f"{row.calibration_violation_error:.4f} | {row.numeric_ks:.4f} | {row.categorical_tv:.4f} |")
    lines += ["", "Selected labels: " + ", ".join(f"{key}=`{value['method']}`" for key, value in selected.items()) + "."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
