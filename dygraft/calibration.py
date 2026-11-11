#!/usr/bin/env python3
"""Select repair prefixes using train-only repeated-fold confidence rules.

The script replays a frozen greedy coverage trace and evaluates only train
reference proxies.  Held-out/test rows are never loaded.  It materializes
three confidence policies plus fixed-2% and a locked gap-15 sensitivity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dygraft import anchoring as bpr
from dygraft.train_coverage import TrainCoverageFold


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(anchor: np.ndarray, trace: list[dict], step: int) -> np.ndarray:
    chosen = set(int(value) for value in anchor)
    for record in trace[:step]:
        remove_index = int(record["remove_index"])
        add_index = int(record["add_index"])
        if remove_index not in chosen or add_index in chosen:
            raise ValueError(f"invalid trace transition at step {record['step']}")
        chosen.remove(remove_index)
        chosen.add(add_index)
    result = np.asarray(sorted(chosen), dtype=np.int64)
    if len(result) != len(anchor):
        raise RuntimeError("trace replay changed cardinality")
    return result


def mean_se(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, se


def earliest_within_one_se(rows: list[dict], score: str,
                           eligible: list[dict]) -> dict:
    best = max(eligible, key=lambda row: (row[f"{score}_mean"], -row["step"]))
    threshold = best[f"{score}_mean"] - best[f"{score}_se"]
    return min(
        (row for row in eligible if row[f"{score}_mean"] + 1e-12 >= threshold),
        key=lambda row: row["step"],
    )


def materialize(name: str, row: dict, *, repo: Path, dataset: str,
                pool: pd.DataFrame, pool_path: Path, anchor_path: Path,
                anchor: np.ndarray, trace: list[dict], repair_metadata: dict,
                output: Path, rule: dict) -> None:
    indices = replay(anchor, trace, int(row["step"]))
    target = output / name
    target.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"pool_index": indices}).to_csv(
        target / "selected_indices.csv", index=False)
    pool.iloc[indices].reset_index(drop=True).to_csv(
        target / "selected.csv", index=False)
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "selection_data": "train_only",
        "policy": name,
        "selected_trace_step": int(row["step"]),
        "selected_swap_fraction": float(row["step"] / len(anchor)),
        "train_proxy": {
            key: value for key, value in row.items() if key != "indices"
        },
        "rule": rule,
        "source_repair_metadata": str(repair_metadata["metadata_path"]),
        "hashes": {
            "pool": sha256(pool_path),
            "anchor": sha256(anchor_path),
            "source_repair_metadata": sha256(repair_metadata["metadata_path"]),
        },
    }
    (target / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--checkpoint-stride", type=int, default=8)
    parser.add_argument(
        "--alpha-floor-drop",
        type=float,
        default=None,
        help=("Optional absolute train-alpha drop from the anchor mean. "
              "When omitted, preserve the original one-standard-error floor."),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.folds < 3 or args.checkpoint_stride < 1 or
            (args.alpha_floor_drop is not None and args.alpha_floor_drop < 0)):
        raise ValueError("invalid confidence-policy parameters")

    metadata_path = args.repair_dir / "selection_metadata.json"
    repair = json.loads(metadata_path.read_text())
    repair["metadata_path"] = metadata_path
    anchor_path = Path(repair["parameters"]["anchor_index"])
    if repair["hashes"]["pool"] != sha256(args.pool):
        raise ValueError("pool hash mismatch")
    if repair["hashes"]["anchor"] != sha256(anchor_path):
        raise ValueError("anchor hash mismatch")

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_path = args.repo / "synthetic" / args.dataset / "real.csv"
    train = bpr.canonicalize(pd.read_csv(train_path), info)
    pool = bpr.canonicalize(pd.read_csv(args.pool), info)
    anchor_frame = pd.read_csv(anchor_path)
    if list(anchor_frame.columns) != ["pool_index"]:
        raise ValueError("invalid anchor schema")
    anchor = anchor_frame["pool_index"].to_numpy(np.int64)
    trace = list(repair["trace"])

    steps = list(range(0, len(trace) + 1, args.checkpoint_stride))
    for special in (len(trace), min(len(trace), int(round(len(anchor) * .01))),
                    min(len(trace), int(round(len(anchor) * .02)))):
        if special not in steps:
            steps.append(special)
    steps = sorted(set(steps))
    fold_seeds = [args.seed + 7919 * (index + 1) for index in range(args.folds)]
    folds = [TrainCoverageFold(train, info, len(anchor), seed) for seed in fold_seeds]
    rows: list[dict] = []
    fold_values: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for step in steps:
        indices = replay(anchor, trace, step)
        synthetic = pool.iloc[indices].reset_index(drop=True)
        scores = np.asarray([
            fold.score(synthetic) for fold in folds
        ], dtype=float)
        alpha, beta = scores[:, 0], scores[:, 1]
        joint = 2.0 * alpha * beta / np.clip(alpha + beta, 1e-12, None)
        alpha_mean, alpha_se = mean_se(alpha)
        beta_mean, beta_se = mean_se(beta)
        joint_mean, joint_se = mean_se(joint)
        rows.append({
            "step": step,
            "swap_fraction": step / len(anchor),
            "alpha_mean": alpha_mean, "alpha_se": alpha_se,
            "beta_mean": beta_mean, "beta_se": beta_se,
            "joint_mean": joint_mean, "joint_se": joint_se,
        })
        fold_values[step] = (alpha, beta)

    anchor_row = next(row for row in rows if row["step"] == 0)
    if args.alpha_floor_drop is None:
        alpha_floor = anchor_row["alpha_mean"] - anchor_row["alpha_se"]
        alpha_floor_rule = "anchor_train_alpha_mean_minus_one_standard_error"
    else:
        alpha_floor = anchor_row["alpha_mean"] - args.alpha_floor_drop
        alpha_floor_rule = "anchor_train_alpha_mean_minus_fixed_drop"
    eligible = [row for row in rows if row["alpha_mean"] + 1e-12 >= alpha_floor]
    if not eligible:
        eligible = [anchor_row]
    beta_row = earliest_within_one_se(rows, "beta", eligible)
    joint_row = earliest_within_one_se(rows, "joint", eligible)

    anchor_beta = fold_values[0][1]
    stable_candidates = []
    for row in eligible:
        wins = int(np.sum(fold_values[row["step"]][1] > anchor_beta + 1e-12))
        row["beta_fold_wins_vs_anchor"] = wins
        if wins >= args.folds - 1:
            stable_candidates.append(row)
    stable_row = (earliest_within_one_se(rows, "beta", stable_candidates)
                  if stable_candidates else anchor_row)

    fixed_step = min(len(trace), int(round(len(anchor) * .02)))
    fixed_row = next(row for row in rows if row["step"] == fixed_step)
    anchor_coverage = float(repair["anchor_geometry_coverage"])
    ceiling = float(repair["pool_geometry_ceiling"])
    target = anchor_coverage + .15 * max(0.0, ceiling - anchor_coverage)
    gap_step = len(trace)
    for record in trace:
        if float(record["geometry_coverage"]) + 1e-12 >= target:
            gap_step = int(record["step"])
            break
    if gap_step not in steps:
        indices = replay(anchor, trace, gap_step)
        synthetic = pool.iloc[indices].reset_index(drop=True)
        scores = np.asarray([
            fold.score(synthetic) for fold in folds
        ], dtype=float)
        alpha, beta = scores[:, 0], scores[:, 1]
        joint = 2.0 * alpha * beta / np.clip(alpha + beta, 1e-12, None)
        gap_row = {
            "step": gap_step, "swap_fraction": gap_step / len(anchor),
            "alpha_mean": mean_se(alpha)[0], "alpha_se": mean_se(alpha)[1],
            "beta_mean": mean_se(beta)[0], "beta_se": mean_se(beta)[1],
            "joint_mean": mean_se(joint)[0], "joint_se": mean_se(joint)[1],
        }
    else:
        gap_row = next(row for row in rows if row["step"] == gap_step)
    gap_coverage = (anchor_coverage if gap_step == 0 else
                    float(trace[gap_step - 1]["geometry_coverage"]))

    args.output.mkdir(parents=True)
    policies = [
        ("cv_beta_1se", beta_row, {
            "type": "earliest_within_one_standard_error_of_best_train_beta",
            "alpha_floor": alpha_floor, "folds": args.folds}),
        ("cv_joint_1se", joint_row, {
            "type": "earliest_within_one_standard_error_of_best_train_harmonic_alpha_beta",
            "alpha_floor": alpha_floor, "folds": args.folds}),
        ("cv_stable", stable_row, {
            "type": "four_of_five_paired_train_beta_wins_then_one_se",
            "required_wins": args.folds - 1, "alpha_floor": alpha_floor}),
        ("fixed020", fixed_row, {"type": "fixed_fraction_reference", "fraction": .02}),
        ("gap15_locked", gap_row, {
            "type": "locked_development_sensitivity", "reachable_gap_fraction": .15,
            "target_geometry_coverage": target,
            "target_reached": gap_coverage + 1e-12 >= target}),
    ]
    for name, row, rule in policies:
        materialize(
            name, row, repo=args.repo, dataset=args.dataset, pool=pool,
            pool_path=args.pool, anchor_path=anchor_path, anchor=anchor,
            trace=trace, repair_metadata=repair, output=args.output, rule=rule)

    pd.DataFrame(rows).sort_values("step").to_csv(
        args.output / "train_proxy_curve.csv", index=False)
    manifest = {
        "selection_data": "train_only",
        "fold_seeds": fold_seeds,
        "alpha_floor": alpha_floor,
        "alpha_floor_rule": alpha_floor_rule,
        "alpha_floor_drop": args.alpha_floor_drop,
        "policies": [{"name": name, "step": int(row["step"]),
                      "swap_fraction": float(row["step"] / len(anchor)),
                      "rule": rule} for name, row, rule in policies],
    }
    (args.output / "policy_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
