#!/usr/bin/env python3
"""Row-wise temporal proposal pools for dynamic graph conditional repainting.

Unlike the fixed-checkpoint baseline, this method observes several predicted-
clean states from the *same* reverse trajectory.  Graph violations determine
which columns may be reopened; conditional diffusion proposals are drawn from
each selected state, and one frozen train-only selector jointly chooses the
proposal and intervention time for every row.  The untouched base row is
always a candidate, so the method can abstain.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditioned_tabdiff_adapter import decode
from dynamic_graph_conditional_repaint import (
    FrozenProposalScore,
    eligible_edges,
    requirement_target,
)
from graph_conditioned_residual_denoiser import (
    calibrated_minimal_final_repair,
    load_backbone,
    seed_all,
)
from subgroup_projection_online import PreviewModel
from typed_dynamic_graph_guidance import TypedActuator, edge_masks, graph_metrics


@torch.no_grad()
def sample_base_and_previews(diffusion, samples: int, checkpoints: list[float], seed: int):
    seed_all(seed)
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num_cur = diffusion.num_schedule.total_noise(times)
    sigma_cat_cur = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num_cur); sigma_num_next[1:] = sigma_num_cur[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat_cur); sigma_cat_next[1:] = sigma_cat_cur[:-1]
    if diffusion.sampler_params["stochastic_sampler"] or diffusion.sampler_params["second_order_correction"]:
        raise ValueError("paired trajectory repainting requires deterministic first-order sampling")
    numeric = torch.randn((samples, diffusion.num_numerical_features), device=device) * sigma_num_cur[-1]
    categorical = diffusion._sample_masked_prior(samples, len(diffusion.num_classes))
    wanted = {int(torch.argmin(torch.abs(times.squeeze(-1) - value)).item()): value for value in checkpoints}
    base_model = diffusion._denoise_fn
    preview_model = PreviewModel(base_model, diffusion.num_classes).to(device).eval()
    diffusion._denoise_fn = preview_model
    previews = {}; actual = {}; started = time.perf_counter()
    try:
        for index in reversed(range(diffusion.num_timesteps)):
            numeric, categorical, _ = diffusion.edm_update(
                numeric, categorical, index, times[index], times[index - 1] if index > 0 else None,
                times[index], sigma_num_cur[index], sigma_num_next[index], sigma_num_cur[index],
                sigma_cat_cur[index], sigma_cat_next[index], sigma_cat_cur[index])
            if index in wanted:
                if preview_model.numeric is None or preview_model.codes is None:
                    raise RuntimeError("preview model did not expose a predicted-clean state")
                label = wanted[index]
                previews[label] = torch.cat(
                    [preview_model.numeric.detach(), preview_model.codes.detach().to(preview_model.numeric.dtype)], 1
                ).cpu()
                actual[label] = float(times[index])
    finally:
        diffusion._denoise_fn = base_model
    if len(previews) != len(wanted):
        raise RuntimeError(f"captured {len(previews)} of {len(wanted)} requested states")
    return torch.cat([numeric, categorical], 1).cpu(), previews, actual, time.perf_counter() - started


def temporal_target_groups(frames: list[pd.DataFrame], edges: list[dict], numeric_names: list[str],
                           categorical_names: list[str], max_targets: int, policy: str):
    priorities = {"hard": 3, "near_hard": 2, "soft": 1}
    per_row: dict[int, list[tuple[int, float, str]]] = defaultdict(list)
    target_times: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    edge_diagnostics = {}; active_rows = [set() for _ in frames]
    for edge in edges:
        target = requirement_target(edge, numeric_names, categorical_names)
        if target is None:
            continue
        bad = [edge_masks(frame, edge)[2] for frame in frames]
        for time_index, mask in enumerate(bad):
            active = np.flatnonzero(mask).tolist()
            active_rows[time_index].update(active)
            for row in active:
                target_times[int(row)][target].add(time_index)
        stacked = np.stack(bad, axis=0)
        if policy == "latest":
            selected = stacked[-1]
        elif policy == "persistent":
            selected = stacked.all(axis=0)
        elif policy in {"union", "routed"}:
            selected = stacked.any(axis=0)
        else:
            raise ValueError(policy)
        edge_diagnostics[str(edge["candidate_id"])] = {
            "per_time": [int(mask.sum()) for mask in bad], "selected": int(selected.sum())}
        priority = priorities.get(edge.get("classification", "soft"), 1)
        confidence = float(edge.get("confidence", 0.0))
        for row in np.flatnonzero(selected):
            per_row[int(row)].append((priority, confidence, target))
    grouped: dict[tuple[tuple[str, ...], tuple[int, ...]], list[int]] = defaultdict(list)
    for row, candidates in per_row.items():
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        targets = []
        for _, _, target in candidates:
            if target not in targets:
                targets.append(target)
            if len(targets) >= max_targets:
                break
        if targets:
            if policy == "routed":
                allowed_times = tuple(sorted(set().union(*(target_times[row][target] for target in targets))))
            else:
                allowed_times = tuple(range(len(frames)))
            grouped[(tuple(sorted(targets)), allowed_times)].append(row)
    diagnostics = {
        "active_rows_per_time": [len(rows) for rows in active_rows],
        "selected_rows": int(len(per_row)), "edge_activity": edge_diagnostics,
    }
    return {key: np.asarray(value, dtype=int) for key, value in grouped.items()}, diagnostics


@torch.no_grad()
def temporal_repaint(diffusion, train_data, info: dict, base_frame: pd.DataFrame,
                     preview_encoded: list[torch.Tensor], preview_frames: list[pd.DataFrame],
                     edges: list[dict], scorer: FrozenProposalScore, branches_per_time: int,
                     rounds: int, max_targets: int, seed: int, accept_margin: float, policy: str):
    numeric_names = scorer.numeric_names; categorical_names = scorer.categorical_names
    groups, activity = temporal_target_groups(
        preview_frames, edges, numeric_names, categorical_names, max_targets, policy)
    result = base_frame.copy(); choices = Counter(); generated = 0; started = time.perf_counter()
    group_rows = {}
    for ordinal, ((targets, allowed_times), indices) in enumerate(sorted(groups.items())):
        num_positions = [numeric_names.index(name) for name in targets if name in numeric_names]
        cat_positions = [categorical_names.index(name) for name in targets if name in categorical_names]
        base_subset = base_frame.iloc[indices].reset_index(drop=True)
        proposals = [base_subset.copy()]; sources = ["base"]
        for time_index, encoded in enumerate(preview_encoded):
            if time_index not in allowed_times:
                continue
            subset = encoded[indices]
            for branch in range(branches_per_time):
                seed_all(seed + ordinal * 10007 + time_index * 1009 + branch)
                sampled = diffusion.sample_impute(
                    subset[:, :train_data.d_numerical].to(diffusion.device),
                    subset[:, train_data.d_numerical:].long().to(diffusion.device),
                    num_positions, cat_positions, rounds, "x_t", 0.0, 0.0)
                decoded = decode(train_data, sampled, info)
                candidate = base_subset.copy()
                for name in targets:
                    candidate[name] = decoded[name].to_numpy()
                proposals.append(candidate); sources.append(f"time{time_index}_branch{branch}")
        scores = np.stack([scorer(base_subset, proposal, targets) for proposal in proposals], 0)
        scores[~np.isfinite(scores)] = np.inf
        choice = np.argmin(scores, axis=0)
        improvement = scores[0] - scores[choice, np.arange(len(indices))]
        choice[improvement < accept_margin] = 0
        for proposal_index, proposal in enumerate(proposals):
            selected = choice == proposal_index
            if not selected.any():
                continue
            choices[sources[proposal_index]] += int(selected.sum())
            if proposal_index:
                destination = indices[selected]
                for name in targets:
                    result.loc[destination, name] = proposal.loc[selected, name].to_numpy()
        generated += branches_per_time * len(allowed_times) * len(indices)
        group_rows[f"{'|'.join(targets)}@{','.join(map(str, allowed_times))}"] = int(len(indices))
    diagnostics = {
        **activity, "policy": policy, "target_groups": len(groups), "group_rows": group_rows,
        "proposal_rows_generated": generated, "proposal_choices": dict(choices),
        "unique_rows_changed": int(sum(value for key, value in choices.items() if key != "base")),
        "runtime_seconds": time.perf_counter() - started,
    }
    return result, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", default=".40,.25")
    parser.add_argument("--include-final-state", action="store_true",
                        help="Add post-generation state to the proposal pool and graph activation sequence.")
    parser.add_argument("--activation-policy", choices=["latest", "persistent", "union", "routed"], default="union")
    parser.add_argument("--samples", type=int, default=2048); parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--branches-per-time", type=int, default=4); parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=3); parser.add_argument("--max-calibration-rate", type=float, default=.05)
    parser.add_argument("--graph-weight", type=float, default=20.0); parser.add_argument("--density-weight", type=float, default=1.0)
    parser.add_argument("--rarity-weight", type=float, default=.1); parser.add_argument("--edit-weight", type=float, default=.5)
    parser.add_argument("--conditional-weight", type=float, default=0.0)
    parser.add_argument("--conditional-neighbors", type=int, default=64)
    parser.add_argument("--accept-margin", type=float, default=0.0); parser.add_argument("--density-bank", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261440)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True); sample_dir = args.output / "samples"; sample_dir.mkdir()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    checkpoints = [float(value) for value in args.checkpoints.split(",")]
    device = torch.device("cuda"); started = time.perf_counter()
    diffusion, train_data, test_data, info, _ = load_backbone(
        args.repo, args.dataset, args.config, args.checkpoint, device, args.steps)
    edges = json.loads(args.graph.read_text()); selected_edges = eligible_edges(edges, args.max_calibration_rate)
    if not selected_edges:
        raise RuntimeError("no graph edge passed the actuation/calibration filter")
    train = decode(train_data, train_data.X, info); heldout = decode(train_data, test_data.X, info)
    base_encoded, previews, actual, base_runtime = sample_base_and_previews(
        diffusion, args.samples, checkpoints, args.seed + 100000)
    base_frame = decode(train_data, base_encoded, info); base_frame.to_csv(sample_dir / "tabdiff.csv", index=False)
    ordered_encoded = [previews[value] for value in checkpoints]
    ordered_frames = [decode(train_data, previews[value], info) for value in checkpoints]
    for value, frame in zip(checkpoints, ordered_frames):
        frame.to_csv(sample_dir / f"trigger_preview_{value:.2f}.csv", index=False)
    if args.include_final_state:
        ordered_encoded.append(base_encoded)
        ordered_frames.append(base_frame)
        base_frame.to_csv(sample_dir / "trigger_preview_final.csv", index=False)
    if args.conditional_weight > 0:
        # Imported lazily to keep the original frozen-selector baseline unchanged.
        from conditional_density_graph_repaint import ConditionalProposalScore
        scorer = ConditionalProposalScore(
            train_data, train, info, selected_edges, args.graph_weight, args.density_weight,
            args.rarity_weight, args.edit_weight, args.density_bank, train_frame=train,
            conditional_weight=args.conditional_weight,
            conditional_neighbors=args.conditional_neighbors)
    else:
        scorer = FrozenProposalScore(
            train_data, train, info, selected_edges, args.graph_weight, args.density_weight,
            args.rarity_weight, args.edit_weight, args.density_bank)
    repaint, diagnostics = temporal_repaint(
        diffusion, train_data, info, base_frame, ordered_encoded, ordered_frames, selected_edges, scorer,
        args.branches_per_time, args.rounds, args.max_targets, args.seed + 200000,
        args.accept_margin, args.activation_policy)
    repaint.to_csv(sample_dir / "graph_repaint_trajectory.csv", index=False)
    actuator = TypedActuator(edges, train_data, train, info)
    base_repair, base_repair_diag = calibrated_minimal_final_repair(base_frame, actuator)
    repaint_repair, repaint_repair_diag = calibrated_minimal_final_repair(repaint, actuator)
    base_repair.to_csv(sample_dir / "calibrated_final_repair.csv", index=False)
    repaint_repair.to_csv(sample_dir / "graph_repaint_trajectory_plus_repair.csv", index=False)
    rows = [
        {"method": "tabdiff", **graph_metrics(base_frame, heldout, edges), "runtime_seconds": base_runtime},
        {"method": "calibrated_final_repair", **graph_metrics(base_repair, heldout, edges),
         "runtime_seconds": base_runtime, **base_repair_diag},
        {"method": "graph_repaint_trajectory", **graph_metrics(repaint, heldout, edges),
         "runtime_seconds": base_runtime + diagnostics["runtime_seconds"], **diagnostics},
        {"method": "graph_repaint_trajectory_plus_repair", **graph_metrics(repaint_repair, heldout, edges),
         "runtime_seconds": base_runtime + diagnostics["runtime_seconds"], **repaint_repair_diag},
    ]
    pd.DataFrame(rows).to_csv(args.output / "mechanism_metrics.csv", index=False)
    metadata = {
        "method": "trajectory_graph_conditional_repainting", "dataset": args.dataset,
        "requested_checkpoints": checkpoints, "actual_checkpoints": actual,
        "include_final_state": args.include_final_state,
        "activation_policy": args.activation_policy, "graph_edges": len(edges),
        "eligible_edges": len(selected_edges), "api_calls": 0, "selection_uses_heldout": False,
        "diagnostics": diagnostics, "runtime_seconds": time.perf_counter() - started,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
