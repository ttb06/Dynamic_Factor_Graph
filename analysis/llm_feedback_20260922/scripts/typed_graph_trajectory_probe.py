#!/usr/bin/env python3
"""Measure when frozen typed graph edges become readable on TabDiff x-hat-0 trajectories."""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditioned_tabdiff_adapter import decode, seed_all
from general_state_graph_probe import confusion, state_stats
from knowledge_graph_protocol import predicate_mask
from subgroup_trajectory_probe import TracePreviewModel, time_bin
from tabdiff_probe import preprocess_dataset


def masks(frame: pd.DataFrame, edge: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.ones(len(frame), dtype=bool)
    for atom in edge.get("applicability", []):
        active &= predicate_mask(frame, atom)
    excluded = np.zeros(len(frame), dtype=bool)
    for group in edge.get("exceptions", []):
        group_mask = np.ones(len(frame), dtype=bool)
        for atom in group.get("all_of", []):
            group_mask &= predicate_mask(frame, atom)
        excluded |= group_mask
    active &= ~excluded
    satisfied = np.ones(len(frame), dtype=bool)
    for atom in edge.get("requirements", []):
        satisfied &= predicate_mask(frame, atom)
    return active, satisfied, active & ~satisfied


def ratio(a: int, b: int) -> float:
    return a / b if b else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="shoppers")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--precision", type=float, default=.90)
    parser.add_argument("--recall", type=float, default=.80)
    parser.add_argument("--min-final-positive", type=int, default=8)
    parser.add_argument("--certifier-min-t", type=float, default=.20)
    parser.add_argument("--certifier-max-t", type=float, default=.55)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")

    edges = json.loads(args.graph.read_text())
    if not edges:
        raise RuntimeError("empty active graph")
    seed_all(args.seed)
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, args.dataset)
    heldout = decode(train_data, test_data.X, info)
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    base = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(checkpoint["denoise_fn"], strict=True)
    traced = TracePreviewModel(base, train_data.categories).cuda()
    dp = copy.deepcopy(config["diffusion_params"])
    dp.update(num_timesteps=args.steps, scheduler="power_mean_per_column",
              cat_scheduler="log_linear_per_column")
    dp["sampler_params"] = copy.deepcopy(dp["sampler_params"])
    dp["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, traced, None,
        **dp, device=torch.device("cuda")).cuda()
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    diffusion.eval()
    traced.start()
    with torch.no_grad():
        final_encoded = diffusion.sample(args.samples)
    history = traced.stop()
    final = decode(train_data, final_encoded, info)
    final_masks = {edge["candidate_id"]: masks(final, edge) for edge in edges}
    heldout_masks = {edge["candidate_id"]: masks(heldout, edge) for edge in edges}

    rows, previous = [], {}
    streak_rows, violation_streaks = [], {}
    certifier_features, certifier_labels, certifier_edges, certifier_samples = [], [], [], []
    for call_index, trace in enumerate(history):
        encoded = torch.cat([trace["numeric"], trace["codes"].float()], dim=1)
        preview = decode(train_data, encoded, info)
        encoded_numpy = encoded.detach().cpu().numpy().astype(np.float32)
        for edge in edges:
            edge_id = edge["candidate_id"]
            active, satisfied, violation = masks(preview, edge)
            final_active, _, final_violation = final_masks[edge_id]
            am, vm = confusion(active, final_active), confusion(violation, final_violation)
            prior = previous.get(edge_id)
            flip = float((active != prior).mean()) if prior is not None else math.nan
            previous[edge_id] = active
            prior_streak = violation_streaks.get(edge_id, np.zeros(len(violation), dtype=np.int16))
            streak = np.where(violation, prior_streak + 1, 0).astype(np.int16)
            violation_streaks[edge_id] = streak
            clipped = np.clip(streak, 0, 8)
            for streak_bin in np.unique(clipped[violation]):
                selected = violation & (clipped == streak_bin)
                support = int(selected.sum())
                positives = int((selected & final_violation).sum())
                streak_rows.append({"candidate_id": edge_id, "call_index": call_index,
                    "t": trace["t"], "streak_bin": int(streak_bin), "support": support,
                    "final_violation_positives": positives,
                    "final_violation_posterior": ratio(positives, support)})
            if (edge.get("usage_role", "actuate") == "actuate" and
                    args.certifier_min_t <= float(trace["t"]) <= args.certifier_max_t):
                selected_indices = np.flatnonzero(violation)
                if len(selected_indices):
                    extra = np.column_stack([
                        np.full(len(selected_indices), float(trace["t"]), dtype=np.float32),
                        np.log1p(streak[selected_indices]).astype(np.float32),
                    ])
                    certifier_features.append(np.concatenate([encoded_numpy[selected_indices], extra], axis=1))
                    certifier_labels.append(final_violation[selected_indices].astype(np.uint8))
                    certifier_edges.append(np.full(len(selected_indices), edge_id, dtype="U64"))
                    certifier_samples.append(selected_indices.astype(np.int32))
            stats = state_stats(active, satisfied)
            rows.append({"candidate_id": edge_id, "description": edge["description"],
                "classification": edge["classification"], "family": edge["relation_family"],
                "source": edge.get("source", "unknown"),
                "usage_role": edge.get("usage_role", "actuate"),
                "call_index": call_index, "t": trace["t"], "time_bin": time_bin(trace["t"]),
                **stats, "active_precision": am["precision"], "active_recall": am["recall"],
                "violation_precision": vm["precision"], "violation_recall": vm["recall"],
                "active_flip": flip, "final_active": int(final_active.sum()),
                "final_violations": int(final_violation.sum())})

    trajectory = pd.DataFrame(rows)
    trajectory.to_csv(args.output / "edge_state_trajectory.csv", index=False)
    pd.DataFrame(streak_rows).to_csv(args.output / "violation_streak_calibration.csv", index=False)
    if certifier_features:
        np.savez_compressed(args.output / "trajectory_certifier_samples.npz",
            features=np.concatenate(certifier_features), labels=np.concatenate(certifier_labels),
            edge_ids=np.concatenate(certifier_edges), sample_ids=np.concatenate(certifier_samples))

    schedules = []
    for edge in edges:
        edge_id = edge["candidate_id"]
        block = trajectory[trajectory.candidate_id.eq(edge_id)].copy()
        violation_evaluable = int(block.final_violations.max()) >= args.min_final_positive
        eligible = block.active_precision.fillna(0).ge(args.precision) & block.active_recall.fillna(0).ge(args.recall)
        if violation_evaluable:
            eligible &= block.violation_precision.fillna(0).ge(args.precision) & block.violation_recall.fillna(0).ge(args.recall)
        good = block[eligible]
        final_active, _, final_violation = final_masks[edge_id]
        heldout_active, _, heldout_violation = heldout_masks[edge_id]
        schedules.append({"candidate_id": edge_id, "description": edge["description"],
            "classification": edge["classification"],
            "cutoff": float(good.t.max()) if not good.empty else math.nan,
            "family": edge["relation_family"], "source": edge.get("source", "unknown"),
            "usage_role": edge.get("usage_role", "actuate"), "readable": not good.empty,
            "violation_evaluable": violation_evaluable, "final_support": int(final_active.sum()),
            "final_violations": int(final_violation.sum()),
            "synthetic_violation_rate": ratio(int(final_violation.sum()), int(final_active.sum())),
            "heldout_violation_rate": ratio(int(heldout_violation.sum()), int(heldout_active.sum())),
            "calibration_violation_rate": edge["calibration"]["violation_rate"]})
    schedule = pd.DataFrame(schedules)
    schedule.to_csv(args.output / "typed_readability_schedule.csv", index=False)
    summary = trajectory.groupby("time_bin", sort=False).agg(
        active_precision=("active_precision", "mean"), active_recall=("active_recall", "mean"),
        violation_precision=("violation_precision", "mean"), violation_recall=("violation_recall", "mean"),
        topology_flip=("active_flip", "mean"), edges=("candidate_id", "nunique")).reset_index()
    summary.to_csv(args.output / "trajectory_summary.csv", index=False)
    group_summary = schedule.groupby(["source", "family", "usage_role"], dropna=False).agg(
        edges=("candidate_id", "size"), readable=("readable", "sum"),
        mean_cutoff=("cutoff", "mean"), min_cutoff=("cutoff", "min"), max_cutoff=("cutoff", "max"),
        final_support=("final_support", "sum"), final_violations=("final_violations", "sum")).reset_index()
    group_summary.to_csv(args.output / "readability_by_source_family.csv", index=False)
    lines = ["# Typed active-graph trajectory probe", "",
        "| Edge | Class | Cutoff | Final support | Final violations | Synthetic/heldout violation |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in schedule.itertuples(index=False):
        cutoff = f"{row.cutoff:.3f}" if np.isfinite(row.cutoff) else "unreadable"
        lines.append(f"| {row.candidate_id} | {row.classification} | {cutoff} | {row.final_support} | {row.final_violations} | {row.synthetic_violation_rate:.4f}/{row.heldout_violation_rate:.4f} |")
    lines += ["", "Cutoff is the earliest reverse-time state (largest t) meeting the requested precision/recall. For edges with too few final violations, cutoff uses applicability readability only."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({"dataset": args.dataset,
        "samples": args.samples, "steps": args.steps, "edges": len(edges), "seed": args.seed,
        "real_rows_emitted": False, "certifier_min_t": args.certifier_min_t,
        "certifier_max_t": args.certifier_max_t}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
