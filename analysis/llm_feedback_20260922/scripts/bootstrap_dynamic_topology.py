#!/usr/bin/env python3
"""Paired Poisson row-bootstrap for dynamic versus frozen middle-step topology."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from typed_dynamic_graph_guidance import edge_masks


DATASETS = {
    "shoppers": {
        "seeds": ["20261031", "20261032", "20261033"],
        "dynamic": "time_seed_shoppers_middle_s{seed}_dynamic_1060317_*",
        "frozen": "time_seed_shoppers_middle_s{seed}_frozen_state_1060317_*",
    },
    "default": {
        "seeds": ["20261031", "20261032", "20261033"],
        "dynamic": "time_seed_default_middle_s{seed}_dynamic_1060317_*",
        "frozen": "time_seed_default_middle_s{seed}_frozen_state_1060317_*",
    },
    "diabetes": {
        "seeds": ["20261026", "20261027", "20261028"],
        "dynamic": "diabetes_middle_s{seed}_dynamic_1060452_*",
        "frozen": "diabetes_middle_s{seed}_frozen_state_1060452_*",
    },
}


def metric_from_counts(
    active: np.ndarray,
    bad: np.ndarray,
    heldout: np.ndarray,
) -> np.ndarray:
    rates = np.divide(bad, active, out=np.full_like(bad, np.nan, dtype=float), where=active > 0)
    return np.nanmean(np.abs(rates - heldout[None, :]), axis=1)


def seed_bootstrap(dynamic_dir: Path, frozen_dir: Path, replicates: int, rng: np.random.Generator) -> dict:
    dynamic = pd.read_csv(dynamic_dir / "samples" / "x0_reward_calibrated.csv")
    frozen = pd.read_csv(frozen_dir / "samples" / "x0_reward_calibrated.csv")
    if len(dynamic) != len(frozen):
        raise ValueError("paired outputs have different row counts")
    edges = json.loads((dynamic_dir / "active_graph.json").read_text())
    edge_table = pd.read_csv(dynamic_dir / "edge_metrics.csv")
    edge_table = edge_table.loc[edge_table.method.eq("x0_reward_calibrated")].set_index("candidate_id")

    edge_ids = [edge["candidate_id"] for edge in edges]
    heldout = np.asarray([edge_table.loc[edge_id, "heldout_violation_rate"] for edge_id in edge_ids], dtype=float)
    roles = np.asarray([edge.get("usage_role", "actuate") for edge in edges])
    matrices = []
    observed = {}
    for name, frame in [("dynamic", dynamic), ("frozen", frozen)]:
        active_columns, bad_columns = [], []
        for edge in edges:
            active, _, bad = edge_masks(frame, edge)
            active_columns.append(active.astype(np.float32))
            bad_columns.append(bad.astype(np.float32))
        active = np.stack(active_columns, axis=1)
        bad = np.stack(bad_columns, axis=1)
        matrices.extend([active, bad])
        observed[name] = {}
        for role in ["actuate", "all"]:
            keep = roles == "actuate" if role == "actuate" else np.ones(len(edges), dtype=bool)
            observed[name][role] = float(metric_from_counts(
                active[:, keep].sum(axis=0, keepdims=True),
                bad[:, keep].sum(axis=0, keepdims=True),
                heldout[keep],
            )[0])

    # Poisson(1) bootstrap retains row-level dependence across edges and methods
    # while avoiding materializing integer resample indices.
    matrix = np.concatenate(matrices, axis=1)
    counts = np.empty((replicates, matrix.shape[1]), dtype=np.float64)
    chunk = 100
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        weights = rng.poisson(1.0, size=(stop - start, len(dynamic))).astype(np.float32)
        counts[start:stop] = weights @ matrix
    edge_count = len(edges)
    da, db, fa, fb = np.split(counts, [edge_count, 2 * edge_count, 3 * edge_count], axis=1)
    bootstrap = {}
    for role in ["actuate", "all"]:
        keep = roles == "actuate" if role == "actuate" else np.ones(edge_count, dtype=bool)
        dynamic_metric = metric_from_counts(da[:, keep], db[:, keep], heldout[keep])
        frozen_metric = metric_from_counts(fa[:, keep], fb[:, keep], heldout[keep])
        bootstrap[role] = frozen_metric - dynamic_metric
    return {"observed": observed, "bootstrap": bootstrap}


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.analysis / "outputs"
    rng = np.random.default_rng(args.seed)
    rows = []
    per_seed_rows = []
    for dataset, config in DATASETS.items():
        seed_results = []
        for seed in config["seeds"]:
            dynamic_dir = next(root.glob(config["dynamic"].format(seed=seed)))
            frozen_dir = next(root.glob(config["frozen"].format(seed=seed)))
            result = seed_bootstrap(dynamic_dir, frozen_dir, args.replicates, rng)
            seed_results.append(result)
            for role in ["actuate", "all"]:
                improvement = result["observed"]["frozen"][role] - result["observed"]["dynamic"][role]
                low, high = interval(result["bootstrap"][role])
                per_seed_rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "role": role,
                    "dynamic_error": result["observed"]["dynamic"][role],
                    "frozen_error": result["observed"]["frozen"][role],
                    "improvement": improvement,
                    "row_bootstrap_low": low,
                    "row_bootstrap_high": high,
                    "row_bootstrap_probability_positive": float(np.mean(result["bootstrap"][role] > 0)),
                })
        for role in ["actuate", "all"]:
            observed_dynamic = np.asarray([value["observed"]["dynamic"][role] for value in seed_results])
            observed_frozen = np.asarray([value["observed"]["frozen"][role] for value in seed_results])
            conditional = np.stack([value["bootstrap"][role] for value in seed_results]).mean(axis=0)
            hierarchical = np.empty(args.replicates, dtype=float)
            for index in range(args.replicates):
                chosen_seeds = rng.integers(0, len(seed_results), size=len(seed_results))
                chosen_replicates = rng.integers(0, args.replicates, size=len(seed_results))
                hierarchical[index] = np.mean([
                    seed_results[s]["bootstrap"][role][r]
                    for s, r in zip(chosen_seeds, chosen_replicates)
                ])
            conditional_low, conditional_high = interval(conditional)
            hierarchical_low, hierarchical_high = interval(hierarchical)
            rows.append({
                "dataset": dataset,
                "role": role,
                "dynamic_error_mean": float(observed_dynamic.mean()),
                "frozen_error_mean": float(observed_frozen.mean()),
                "improvement_mean": float((observed_frozen - observed_dynamic).mean()),
                "seed_win_rate": float(np.mean(observed_dynamic < observed_frozen)),
                "conditional_row_bootstrap_low": conditional_low,
                "conditional_row_bootstrap_high": conditional_high,
                "hierarchical_bootstrap_low": hierarchical_low,
                "hierarchical_bootstrap_high": hierarchical_high,
                "hierarchical_probability_positive": float(np.mean(hierarchical > 0)),
            })

    summary = pd.DataFrame(rows)
    per_seed = pd.DataFrame(per_seed_rows)
    summary.to_csv(args.output / "bootstrap_summary.csv", index=False)
    per_seed.to_csv(args.output / "bootstrap_per_seed.csv", index=False)
    display = summary.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6f}")
    header = "| " + " | ".join(map(str, display.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    table_rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    lines = [
        "# Paired bootstrap: dynamic versus frozen middle topology",
        "",
        "Positive improvement means lower held-out violation error for dynamic topology.",
        "The conditional interval resamples synthetic rows within the three fixed seeds; the hierarchical interval also resamples seeds.",
        "",
        header,
        divider,
        *table_rows,
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
