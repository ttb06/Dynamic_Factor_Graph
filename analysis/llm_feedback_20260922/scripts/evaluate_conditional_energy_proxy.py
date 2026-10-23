#!/usr/bin/env python3
"""Evaluate row-gated per-step energy controls for conditional requests."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from evaluate_conditional_topology_controls import audit_sample


DIR_RE = re.compile(
    r"conditional_energy_(?P<request>range|category|both)_"
    r"(?P<gate>dynamic|frozen|global)_(?P<job>\d+)_(?P<task>\d+)"
)


def fidelity(frame: pd.DataFrame, heldout: pd.DataFrame, info: dict) -> tuple[float, float]:
    numeric = [info["column_names"][index] for index in info["num_col_idx"]]
    categorical = [info["column_names"][index] for index in info["cat_col_idx"]]
    ks = np.mean([
        ks_2samp(pd.to_numeric(frame[column], errors="coerce"),
                 pd.to_numeric(heldout[column], errors="coerce")).statistic
        for column in numeric
    ])
    tv = []
    for column in categorical:
        left = frame[column].astype(str).value_counts(normalize=True)
        right = heldout[column].astype(str).value_counts(normalize=True)
        keys = left.index.union(right.index)
        tv.append(.5 * float((left.reindex(keys, fill_value=0) - right.reindex(keys, fill_value=0)).abs().sum()))
    return float(ks), float(np.mean(tv))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    info = json.loads((args.repo / "data/default/info.json").read_text())
    heldout = pd.read_csv(args.repo / "synthetic/default/test.csv")
    rows = []
    for directory in sorted(args.root.glob(f"conditional_energy_*_{args.job_id}_*")):
        match = DIR_RE.fullmatch(directory.name)
        if not match:
            continue
        request, gate = match.group("request"), match.group("gate")
        base = pd.read_csv(directory / "samples/unconditional.csv")
        for sample_path in sorted((directory / "samples").glob("proxy_*.csv")):
            sample = pd.read_csv(sample_path)
            ks, tv = fidelity(sample, heldout, info)
            trace_path = directory / f"trace_{sample_path.stem}.json"
            trace = json.loads(trace_path.read_text())
            rows.append({
                "request": request, "gate": gate, "method": sample_path.stem,
                **audit_sample(base, sample, request),
                "numeric_ks": ks, "categorical_tv": tv,
                "topology_flips": trace["topology_flips"],
                "guided_row_steps": trace["guided_row_steps"],
                "runtime_seconds": trace["runtime_seconds"],
                "directory": str(directory),
            })
    result = pd.DataFrame(rows).sort_values(["request", "gate", "method"])
    result.to_csv(args.output / "conditional_energy_metrics.csv", index=False)
    lines = [
        "# Conditional per-step energy proxy", "",
        "Harpoon-style energy is applied only to the active row set inferred from",
        "the predicted-clean categorical state. This is a proxy baseline, not an",
        "official Harpoon reproduction.", "",
        "| Request | Gate | Strength | Violation in scope | Scope loss | Outside-target edits | Numeric KS | Categorical TV | Topology flips |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.request} | {row.gate} | {row.method} | "
            f"{row.violation_given_applicable:.2%} | "
            f"{row.antecedent_evasion_given_base_applicable:.2%} | "
            f"{row.target_changed_given_base_nonapplicable:.2%} | "
            f"{row.numeric_ks:.4f} | {row.categorical_tv:.4f} | {int(row.topology_flips)} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
