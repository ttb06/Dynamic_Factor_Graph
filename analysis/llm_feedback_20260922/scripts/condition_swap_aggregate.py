#!/usr/bin/env python3
"""Aggregate paired-noise condition-vector causal probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value): values.append("")
            elif isinstance(value, float): values.append(f"{value:.4f}")
            else: values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-ids", required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    wanted = {token for token in args.job_ids.replace(",", ":").split(":") if token}
    frames, runs = [], []
    for path in sorted(args.run_root.glob("condition_swap_*")):
        if not any(path.name.endswith(f"_{job_id}") for job_id in wanted): continue
        metric, meta = path / "swap_metrics.csv", path / "run_metadata.json"
        if metric.exists(): frames.append(pd.read_csv(metric))
        if meta.exists(): runs.append(json.loads(meta.read_text()))
    if not frames: raise RuntimeError("no completed swap probes")
    metrics = pd.concat(frames, ignore_index=True); metrics.to_csv(args.output / "all_swap_metrics.csv", index=False)
    summary = metrics.groupby(["representation", "variant"]).agg(
        pairs=("pair", "size"), semantic_margin=("semantic_margin_high_minus_low", "mean"),
        margin_std=("semantic_margin_high_minus_low", "std"), worst_pair_margin=("semantic_margin_high_minus_low", "min"),
        positive_direction_pairs=("semantic_margin_high_minus_low", lambda values: int((values > 0).sum())),
        strict_monotonic=("paired_strict_monotonic_rate", "mean"),
        non_decreasing=("paired_non_decreasing_rate", "mean"), high_satisfaction=("high_request_satisfaction", "mean"),
        other_numeric_change=("mean_other_numeric_change_std_units", "mean"), categorical_change=("categorical_change_rate", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    correct = summary[summary.variant.eq("correct")][["representation", "semantic_margin", "strict_monotonic", "high_satisfaction"]]
    lines = [
        "# Paired-noise condition-semantic audit", "", "## All controls", "", table(summary), "",
        "## Correct conditions", "", table(correct), "",
        "A condition is causally meaningful only if correct high-threshold vectors move the paired output upward, swapped vectors reverse the margin, and null/generic vectors collapse the paired difference. Magnitude matters: a positive but tiny margin is evidence of use, not calibrated control.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({"status": "executed", "runs": runs}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
