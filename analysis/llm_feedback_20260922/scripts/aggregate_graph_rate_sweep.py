#!/usr/bin/env python3
"""Aggregate the one-seed graph-rate controller selection sweep."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHOD = "xstate_reward_calibrated"


def load_run(path: Path, label: str) -> tuple[dict, pd.DataFrame]:
    metrics = pd.read_csv(path / "method_metrics.csv")
    row = metrics.loc[metrics.method.eq(METHOD)].iloc[0].to_dict()
    edges = pd.read_csv(path / "edge_metrics.csv")
    edges = edges.loc[edges.method.eq(METHOD)].copy()
    graph = json.loads((path / "evaluation_graph.json").read_text())
    targets = {}
    for edge in graph:
        rates = [float(edge[name]["violation_rate"])
                 for name in ("discovery", "counterexample", "calibration") if name in edge]
        targets[edge["candidate_id"]] = float(np.median(rates))
    edges["split_median_target"] = edges.candidate_id.map(targets)
    edges["split_median_error"] = (edges.violation_rate - edges.split_median_target).abs()
    edges.insert(0, "label", label)
    row.update({"label": label, "path": str(path),
                "split_median_error": float(edges.split_median_error.mean())})
    return row, edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    edge_frames = []
    baseline_row, baseline_edges = load_run(args.baseline, "split_median_severity")
    rows.append(baseline_row)
    edge_frames.append(baseline_edges)
    pattern = re.compile(r"graph_rate_(.+)_\d+_\d+$")
    for path in args.runs:
        match = pattern.search(path.name)
        if not match:
            raise ValueError(f"cannot parse label from {path}")
        row, edges = load_run(path, match.group(1))
        rows.append(row)
        edge_frames.append(edges)
    frame = pd.DataFrame(rows)
    base_error = float(baseline_row["heldout_violation_error"])
    frame["gain_vs_split_median_pct"] = 100.0 * (base_error - frame.heldout_violation_error) / base_error
    frame = frame.sort_values("heldout_violation_error")
    frame.to_csv(args.output / "method_comparison.csv", index=False)
    pd.concat(edge_frames, ignore_index=True).to_csv(args.output / "edge_comparison.csv", index=False)

    columns = ["label", "heldout_violation_error", "calibration_violation_error",
               "split_median_error", "macro_violation", "numeric_ks", "categorical_tv",
               "action_events", "runtime_seconds", "gain_vs_split_median_pct"]
    lines = ["# Graph-rate controller selection", "",
             "Single paired seed; every target and controller decision uses train partitions only.", "",
             "| Method | Held-out error ↓ | Calibration error ↓ | Split-median error ↓ | Macro violation | KS ↓ | TV ↓ | Actions | Runtime (s) | Gain vs prior best |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in frame[columns].itertuples(index=False):
        lines.append(f"| {row.label} | {row.heldout_violation_error:.6f} | "
                     f"{row.calibration_violation_error:.6f} | {row.split_median_error:.6f} | "
                     f"{row.macro_violation:.6f} | {row.numeric_ks:.6f} | {row.categorical_tv:.6f} | "
                     f"{int(row.action_events)} | {row.runtime_seconds:.2f} | "
                     f"{row.gain_vs_split_median_pct:+.1f}% |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
