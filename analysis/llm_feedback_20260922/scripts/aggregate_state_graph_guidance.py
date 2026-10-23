#!/usr/bin/env python3
"""Aggregate the same-graph, same-actuator dynamic topology experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--pattern", default="state_graph_guidance_*_*/method_metrics.csv")
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(a.input_root.glob(a.pattern))
    if not paths: raise FileNotFoundError("state graph guidance metrics")
    data = pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in paths], ignore_index=True)
    data.to_csv(a.output/"all_method_metrics.csv", index=False)
    baseline = data[data.method.eq("unconditional")][["dataset", "macro_violation", "micro_violation", "numeric_ks", "categorical_tv"]]
    paired = data.merge(baseline, on="dataset", suffixes=("", "_base"), validate="many_to_one")
    for metric in ["macro_violation", "micro_violation", "numeric_ks", "categorical_tv"]:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_base"]
    paired.to_csv(a.output/"paired_to_unconditional.csv", index=False)
    lines = ["# State-conditioned typed graph guidance", "",
             "All online methods use the same verified graph, paired initial noise and typed minimal-projection actuator.", "",
             "| Dataset | Method | Hard/near violation | Calibration error | Conditional distance | Numeric KS | Categorical TV | Row actions | Final repair |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in data.sort_values(["dataset", "method"]).itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.method} | {row.hard_near_violation:.4f} | {row.violation_calibration_error:.4f} | "
                     f"{row.conditional_endpoint_distance:.4f} | {row.numeric_ks:.4f} | {row.categorical_tv:.4f} | "
                     f"{int(row.row_actions)} | {row.final_repair_rate:.4f} |")
    lines += ["", "## Reading rule", "",
              "- Hard/near-hard violation, conditional distance, KS and TV are lower-is-better.",
              "- Calibration error prevents a probabilistic edge from being rewarded merely for being forced to zero violation.",
              "- `fixed_middle` and `dynamic_middle` differ only in state/persistence gating.",
              "- `oracle_middle` uses paired unconditional final violations only as an upper-bound diagnostic.",
              "- `dynamic_middle_plus_certifier` tests whether online guidance reduces residual final repair."]
    (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__": main()
