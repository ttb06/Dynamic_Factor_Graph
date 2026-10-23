#!/usr/bin/env python3
"""Aggregate paired graph-rate confirmation against existing controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ["macro_violation", "heldout_violation_error", "calibration_violation_error",
           "numeric_ks", "categorical_tv", "action_events", "runtime_seconds"]


def collect(path: Path, methods: dict[str, str]) -> tuple[list[dict], list[pd.DataFrame]]:
    metadata = json.loads((path / "run_metadata.json").read_text())
    seed = int(metadata["seed"])
    table = pd.read_csv(path / "method_metrics.csv")
    edge_table = pd.read_csv(path / "edge_metrics.csv")
    rows = []
    edges = []
    for source, label in methods.items():
        row = table.loc[table.method.eq(source)].iloc[0]
        rows.append({"seed": seed, "method": label,
                     **{metric: float(row[metric]) for metric in METRICS}})
        selected = edge_table.loc[edge_table.method.eq(source)].copy()
        selected.insert(0, "seed", seed)
        selected.insert(1, "method_label", label)
        edges.append(selected)
    return rows, edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--graph-rate-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    edges = []
    for path in args.baseline_runs:
        new_rows, new_edges = collect(path, {
            "unconditional": "continued_unguided",
            "xstate_reward_calibrated": "split_median_local",
            "final_reward_calibrated": "split_median_final_repair",
        })
        rows.extend(new_rows)
        edges.extend(new_edges)
    for path in args.graph_rate_runs:
        new_rows, new_edges = collect(path, {"xstate_reward_calibrated": "split_median_graph_rate"})
        rows.extend(new_rows)
        edges.extend(new_edges)
    frame = pd.DataFrame(rows).sort_values(["seed", "method"])
    frame.to_csv(args.output / "per_seed_metrics.csv", index=False)
    pd.concat(edges, ignore_index=True).to_csv(args.output / "per_edge_metrics.csv", index=False)
    summary = frame.groupby("method")[METRICS].agg(["mean", "std"])
    summary.to_csv(args.output / "summary_metrics.csv")
    pivot = frame.pivot(index="seed", columns="method", values="heldout_violation_error")
    gains = pd.DataFrame(index=pivot.index)
    for control in ("continued_unguided", "split_median_local", "split_median_final_repair"):
        gains[f"gain_vs_{control}_pct"] = 100.0 * (
            pivot[control] - pivot["split_median_graph_rate"]
        ) / pivot[control]
    gains.reset_index().to_csv(args.output / "paired_gains.csv", index=False)

    lines = ["# Graph-rate controller: three-seed paired confirmation", "",
             "All runs use the continued-training checkpoint, the same graph, 4,096 rows and paired seeds.", "",
             "| Method | Held-out error ↓ | Calibration error ↓ | Macro violation | KS ↓ | TV ↓ | Actions | Runtime (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method, row in summary.iterrows():
        lines.append(f"| {method} | {row[('heldout_violation_error','mean')]:.6f} ± {row[('heldout_violation_error','std')]:.6f} | "
                     f"{row[('calibration_violation_error','mean')]:.6f} | {row[('macro_violation','mean')]:.6f} | "
                     f"{row[('numeric_ks','mean')]:.6f} | {row[('categorical_tv','mean')]:.6f} | "
                     f"{row[('action_events','mean')]:.1f} | {row[('runtime_seconds','mean')]:.2f} |")
    lines += ["", "| Paired comparison | Mean gain | Seeds won |",
              "| --- | ---: | ---: |"]
    for control in ("continued_unguided", "split_median_local", "split_median_final_repair"):
        column = f"gain_vs_{control}_pct"
        lines.append(f"| graph-rate vs {control} | {gains[column].mean():+.1f}% | "
                     f"{int((gains[column] > 0).sum())}/{len(gains)} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
