#!/usr/bin/env python3
"""Paired Poisson row-bootstrap for held-out graph-error differences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from typed_dynamic_graph_guidance import edge_masks


def edge_arrays(frame: pd.DataFrame, edges: list[dict]):
    return [(edge_masks(frame, edge)[0].astype(float), edge_masks(frame, edge)[2].astype(float))
            for edge in edges]


def weighted_error(arrays, heldout_rates: np.ndarray, weight: np.ndarray) -> float:
    rates = []
    for active, bad in arrays:
        support = float(weight @ active)
        rates.append(float(weight @ bad) / support if support else np.nan)
    rates = np.asarray(rates, dtype=float)
    return float(np.nanmean(np.abs(rates - heldout_rates)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True); parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True); parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-name", default="candidate"); parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--replicates", type=int, default=5000); parser.add_argument("--seed", type=int, default=20261441)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    edges = json.loads(args.graph.read_text()); heldout = pd.read_csv(args.heldout)
    candidate = pd.read_csv(args.candidate); baseline = pd.read_csv(args.baseline)
    if len(candidate) != len(baseline):
        raise ValueError("paired bootstrap requires equal row counts")
    heldout_rates = []
    for edge in edges:
        active, _, bad = edge_masks(heldout, edge)
        heldout_rates.append(float(bad.sum() / active.sum()) if active.sum() else np.nan)
    heldout_rates = np.asarray(heldout_rates)
    candidate_arrays = edge_arrays(candidate, edges); baseline_arrays = edge_arrays(baseline, edges)
    one = np.ones(len(candidate), dtype=float)
    observed_candidate = weighted_error(candidate_arrays, heldout_rates, one)
    observed_baseline = weighted_error(baseline_arrays, heldout_rates, one)
    rng = np.random.default_rng(args.seed); effects = np.empty(args.replicates, dtype=float)
    for replicate in range(args.replicates):
        weight = rng.poisson(1.0, len(candidate)).astype(float)
        effects[replicate] = (weighted_error(baseline_arrays, heldout_rates, weight)
                              - weighted_error(candidate_arrays, heldout_rates, weight))
    summary = {
        "candidate": args.candidate_name, "baseline": args.baseline_name,
        "rows": len(candidate), "edges": len(edges), "replicates": args.replicates,
        "candidate_error": observed_candidate, "baseline_error": observed_baseline,
        "observed_absolute_gain": observed_baseline - observed_candidate,
        "observed_relative_gain_percent": 100 * (observed_baseline - observed_candidate) / max(observed_baseline, 1e-12),
        "ci_low": float(np.quantile(effects, .025)), "ci_high": float(np.quantile(effects, .975)),
        "probability_positive": float(np.mean(effects > 0)),
        "scope": "paired row uncertainty conditional on one sampling seed and a fixed evaluation graph",
    }
    pd.DataFrame({"baseline_minus_candidate": effects}).to_csv(args.output / "bootstrap_draws.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Paired graph-error bootstrap", "", f"- Candidate: `{args.candidate_name}`",
             f"- Baseline: `{args.baseline_name}`",
             f"- Absolute gain: `{summary['observed_absolute_gain']:.6f}`",
             f"- Relative gain: `{summary['observed_relative_gain_percent']:.2f}%`",
             f"- 95% row-bootstrap interval: `[{summary['ci_low']:.6f}, {summary['ci_high']:.6f}]`",
             f"- P(gain > 0): `{summary['probability_positive']:.4f}`", "",
             "This interval does not include sampling-seed or graph-discovery uncertainty."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
