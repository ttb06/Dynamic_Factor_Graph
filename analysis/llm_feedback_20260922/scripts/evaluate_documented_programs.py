#!/usr/bin/env python3
"""Independent audit for typed, externally documented functional programs.

This audit is intentionally separate from sampling.  It reports validity only
for programs backed by dataset metadata; LLM/statistical hypotheses do not
belong in this metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def residual(frame: pd.DataFrame, program: dict) -> tuple[np.ndarray, np.ndarray]:
    active = np.ones(len(frame), dtype=bool)
    if program.get("gate_sum_columns"):
        gate = frame[program["gate_sum_columns"]].apply(
            pd.to_numeric, errors="coerce").fillna(0).to_numpy(float).sum(axis=1)
        active &= gate > float(program.get("gate_sum_gt", 0.0))
    kind = program["kind"]
    if kind == "sum_equal":
        values = frame[program["columns"]].apply(
            pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        error = np.abs(values.sum(axis=1) - float(program["target"]))
    elif kind == "abs_equal":
        source = pd.to_numeric(frame[program["source"]], errors="coerce").fillna(0).to_numpy(float)
        target = pd.to_numeric(frame[program["target_column"]], errors="coerce").fillna(0).to_numpy(float)
        error = np.abs(target - np.abs(source - float(program.get("center", 0.0))))
    elif kind == "ordered_triplet":
        values = frame[program["columns"]].apply(
            pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        error = np.maximum(values[:, :-1] - values[:, 1:], 0).max(axis=1)
    else:
        raise ValueError(kind)
    return active, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--sample", action="append", default=[], help="name=csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    edges = json.loads(args.graph.read_text())
    rows = []
    for specification in args.sample:
        name, path = specification.split("=", 1)
        frame = pd.read_csv(path)
        per_edge = []
        for edge in edges:
            program = edge["program"]
            active, error = residual(frame, program)
            tolerance = float(program.get("tolerance", 1e-6))
            support = int(active.sum())
            violations = int((active & (error > tolerance)).sum())
            selected = error[active]
            row = {
                "method": name, "candidate_id": edge["candidate_id"],
                "support": support, "violations": violations,
                "violation_rate": violations / support if support else np.nan,
                "mean_absolute_residual": float(selected.mean()) if support else np.nan,
                "p95_absolute_residual": float(np.quantile(selected, .95)) if support else np.nan,
            }
            rows.append(row); per_edge.append(row)
        rates = [row["violation_rate"] for row in per_edge]
        rows.append({
            "method": name, "candidate_id": "__macro__",
            "support": int(sum(row["support"] for row in per_edge)),
            "violations": int(sum(row["violations"] for row in per_edge)),
            "violation_rate": float(np.mean(rates)),
            "mean_absolute_residual": float(np.mean([row["mean_absolute_residual"] for row in per_edge])),
            "p95_absolute_residual": np.nan,
        })
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "documented_program_metrics.csv", index=False)
    macro = result[result.candidate_id == "__macro__"]
    # Keep the evaluator self-contained on lean Slurm environments where the
    # optional pandas ``tabulate`` dependency is not installed.
    lines = ["# Documented functional-program audit", "",
             "Only dataset-documented programs are included; discovered soft relations are excluded.", "",
             "```text", macro.to_string(index=False), "```", "", "## Per program", "",
             "```text", result.to_string(index=False), "```"]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
