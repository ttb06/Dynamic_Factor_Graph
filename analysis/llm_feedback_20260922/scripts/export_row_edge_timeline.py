#!/usr/bin/env python3
"""Export representative per-row, per-edge states along a TabDiff trajectory."""
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
from subgroup_trajectory_probe import TracePreviewModel
from tabdiff_probe import preprocess_dataset
from typed_graph_trajectory_probe import masks


STATE_NAMES = ["unreadable", "inactive", "satisfied", "violated", "trigger_ready"]


def native(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return round(float(value), 4)
    if pd.isna(value): return None
    return str(value) if not isinstance(value, (int, float, bool)) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261440)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--active-min-t", type=float, default=0.0)
    parser.add_argument("--active-max-t", type=float, default=0.65)
    parser.add_argument("--volatility-threshold", type=float, default=0.15)
    parser.add_argument("--volatility-ema", type=float, default=0.8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")

    graph = json.loads(args.graph.read_text())
    schedule_frame = pd.read_csv(args.schedule)
    cutoffs = {str(row.candidate_id): float(row.cutoff) for row in schedule_frame.itertuples(index=False)
               if bool(row.readable) and np.isfinite(float(row.cutoff))}
    graph = [edge for edge in graph if edge["candidate_id"] in cutoffs]
    seed_all(args.seed); sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, _ = preprocess_dataset(args.repo, args.dataset)
    with args.config.open("rb") as handle: config = pickle.load(handle)
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
    diffusion.eval(); traced.start()
    with torch.no_grad(): final_encoded = diffusion.sample(args.samples)
    history = traced.stop(); final = decode(train_data, final_encoded, info)

    steps, rows, edges = len(history), args.samples, len(graph)
    states = np.zeros((steps, rows, edges), dtype=np.uint8)
    raw_active_cube = np.zeros((steps, rows, edges), dtype=bool)
    accepted_active_cube = np.zeros_like(raw_active_cube)
    streak_cube = np.zeros((steps, rows, edges), dtype=np.uint8)
    required = np.zeros(steps, dtype=np.uint8)
    accepted = {}; raw_previous = {}; volatility = {}; mismatch_streak = {}; violation_streak = {}
    decoded_history = []

    for step_index, trace in enumerate(history):
        frame = decode(train_data, torch.cat([trace["numeric"], trace["codes"].float()], dim=1), info)
        decoded_history.append(frame)
        t = float(trace["t"])
        width = max(args.active_max_t - args.active_min_t, 1e-8)
        remaining = float(np.clip((t - args.active_min_t) / width, 0.0, 1.0))
        required_streak = max(1, int(math.ceil(1.0 + 7.0 * remaining)))
        required[step_index] = required_streak
        for edge_index, edge in enumerate(graph):
            edge_id = edge["candidate_id"]
            if t > args.active_max_t or t < args.active_min_t or t > cutoffs[edge_id]:
                continue
            raw_active, satisfied, _ = masks(frame, edge)
            raw_active_cube[step_index, :, edge_index] = raw_active
            previous_raw = raw_previous.get(edge_id)
            old_volatility = volatility.get(edge_id, np.zeros(rows, dtype=np.float32))
            flip = ((raw_active != previous_raw).astype(np.float32) if previous_raw is not None
                    else np.zeros(rows, dtype=np.float32))
            row_volatility = (args.volatility_ema * old_volatility +
                              (1.0 - args.volatility_ema) * flip).astype(np.float32)
            volatility[edge_id] = row_volatility; raw_previous[edge_id] = raw_active.copy()
            persistence = np.where(row_volatility >= args.volatility_threshold, 3, 1).astype(np.int16)
            if edge_id not in accepted:
                accepted[edge_id] = raw_active.copy()
                mismatch_streak[edge_id] = np.zeros(rows, dtype=np.int16)
            else:
                mismatch = raw_active != accepted[edge_id]
                local_streak = np.where(mismatch, mismatch_streak[edge_id] + 1, 0).astype(np.int16)
                commit = local_streak >= persistence
                accepted[edge_id][commit] = raw_active[commit]
                local_streak[commit] = 0; mismatch_streak[edge_id] = local_streak
            active = accepted[edge_id].copy()
            accepted_active_cube[step_index, :, edge_index] = active
            violated = active & ~satisfied
            old_streak = violation_streak.get(edge_id, np.zeros(rows, dtype=np.int16))
            current_streak = np.where(violated, old_streak + 1, 0).astype(np.int16)
            violation_streak[edge_id] = current_streak
            streak_cube[step_index, :, edge_index] = np.clip(current_streak, 0, 255)
            state = np.where(~active, 1, np.where(satisfied, 2,
                np.where(current_streak >= required_streak, 4, 3))).astype(np.uint8)
            states[step_index, :, edge_index] = state

    transition_count = np.zeros(rows, dtype=int)
    for edge_index in range(edges):
        block = states[:, :, edge_index]
        transition_count += np.sum((block[1:] != block[:-1]) & (block[1:] > 0) & (block[:-1] > 0), axis=0)
    ready_count = np.sum(states == 4, axis=(0, 2))
    violated_count = np.sum(states == 3, axis=(0, 2))
    active_edge_count = np.sum(np.any(states >= 2, axis=0), axis=1)
    score = 8 * ready_count + 3 * transition_count + violated_count + active_edge_count

    chosen = []
    for edge_index in range(edges):
        edge_score = 10 * np.sum(states[:, :, edge_index] == 4, axis=0) + \
                     3 * np.sum(states[:, :, edge_index] == 3, axis=0)
        for row_index in np.argsort(-edge_score):
            if edge_score[row_index] <= 0: break
            if int(row_index) not in chosen:
                chosen.append(int(row_index)); break
        if len(chosen) >= args.examples: break
    for row_index in np.argsort(-score):
        if len(chosen) >= args.examples: break
        if int(row_index) not in chosen: chosen.append(int(row_index))

    involved_columns = []
    for edge in graph:
        for atom in edge.get("applicability", []) + edge.get("requirements", []):
            for column in (atom.get("left_column"), atom.get("right_column") if atom.get("right_kind") == "column" else None):
                if column and column not in involved_columns: involved_columns.append(column)

    examples = []
    for example_index, row_index in enumerate(chosen):
        timeline = []
        for step_index, (trace, frame) in enumerate(zip(history, decoded_history)):
            edge_states = []
            for edge_index, edge in enumerate(graph):
                code = int(states[step_index, row_index, edge_index])
                edge_states.append({
                    "edge_id": edge["candidate_id"], "state": STATE_NAMES[code],
                    "raw_active": bool(raw_active_cube[step_index, row_index, edge_index]),
                    "stable_active": bool(accepted_active_cube[step_index, row_index, edge_index]),
                    "streak": int(streak_cube[step_index, row_index, edge_index]),
                })
            timeline.append({
                "step": step_index, "t": round(float(trace["t"]), 4),
                "required_streak": int(required[step_index]),
                "values": {column: native(frame.iloc[row_index][column]) for column in involved_columns},
                "edges": edge_states,
            })
        examples.append({
            "label": f"S{example_index + 1}", "sample_index": row_index,
            "selection_score": int(score[row_index]),
            "ready_states": int(ready_count[row_index]),
            "state_transitions": int(transition_count[row_index]),
            "active_edges": int(active_edge_count[row_index]),
            "final_values": {column: native(final.iloc[row_index][column]) for column in involved_columns},
            "timeline": timeline,
        })

    aggregate = []
    for step_index, trace in enumerate(history):
        counts = {name: int((states[step_index] == code).sum()) for code, name in enumerate(STATE_NAMES)}
        aggregate.append({"step": step_index, "t": round(float(trace["t"]), 4),
                          "required_streak": int(required[step_index]), **counts})
    payload = {
        "dataset": args.dataset, "seed": args.seed, "samples": args.samples, "steps": steps,
        "state_names": STATE_NAMES, "active_window": [args.active_min_t, args.active_max_t],
        "topology_mode": "row_ema_hysteresis", "state_gate": "annealed8",
        "graph": [{"edge_id": edge["candidate_id"], "description": edge["description"],
                   "classification": edge["classification"], "cutoff": round(cutoffs[edge["candidate_id"]], 4)}
                  for edge in graph],
        "involved_columns": involved_columns, "aggregate": aggregate, "examples": examples,
        "note": "Examples are replayed unguided calibration trajectories. trigger_ready means the row-edge passes readability, stabilized applicability, and annealed streak gates; it is not an observed action event from the guided run.",
    }
    (args.output / "row_edge_timeline.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(aggregate).to_csv(args.output / "aggregate_state_counts.csv", index=False)
    (args.output / "run_metadata.json").write_text(json.dumps({
        "dataset": args.dataset, "seed": args.seed, "samples": args.samples,
        "selected_sample_indices": chosen, "real_rows_emitted": False,
    }, indent=2))
    print(json.dumps({"selected": chosen, "output": str(args.output), "edges": edges}, indent=2))


if __name__ == "__main__":
    main()
