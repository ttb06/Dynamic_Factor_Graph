#!/usr/bin/env python3
"""Paired mid-trajectory branch-and-lookahead for a frozen TabDiff graph."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditioned_tabdiff_adapter import decode, seed_all
from rowwise_reverse_action_guidance import (
    RowActionLayer,
    calibrated_gate,
    edge_masks,
    edge_metric_rows,
    evaluate,
    final_action,
)
from subgroup_projection_online import PreviewModel
from tabdiff_probe import preprocess_dataset


def get_rng_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state()


def set_rng_state(state) -> None:
    torch.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1])


def preview_step(diffusion, preview, numeric, categorical, index, times, sn, sc):
    samples = len(numeric); sigma = sn[index]
    numeric_hat = numeric + (sigma**2 - sigma**2).sqrt() * torch.randn_like(numeric)
    move_chance = -torch.expm1(sc[index] - sc[index])
    categorical_hat, _ = diffusion.q_xt(categorical, move_chance)
    one_hot = diffusion.to_one_hot(categorical_hat).to(numeric.dtype)
    denoised, raw_logits = preview.base(
        numeric_hat.float(), one_hot, times[index].squeeze().repeat(samples),
        sigma=sigma.unsqueeze(0).repeat(samples, 1),
    )
    codes = preview.predicted_codes(one_hot, raw_logits)
    return numeric_hat, categorical_hat, denoised, raw_logits, codes


def apply_update(diffusion, layer, numeric_hat, categorical_hat, denoised, raw_logits,
                 corrected, changed_numeric, changed_cat, sigma, sigma_next,
                 cat_sigma, cat_sigma_next, mode, rollback_generator):
    device = numeric_hat.device; samples = len(numeric_hat)
    corrected_num = denoised.clone()
    if changed_numeric.any() and mode != "rollback":
        encoded = torch.as_tensor(layer.encode_numeric(corrected), device=device, dtype=numeric_hat.dtype)
        mask = torch.as_tensor(changed_numeric, device=device)
        corrected_num[mask] = encoded[mask]
    direction = (numeric_hat - corrected_num) / sigma
    numeric_next = numeric_hat + (sigma_next - sigma) * direction
    logits = diffusion._subs_parameterization(raw_logits, categorical_hat)
    alpha_t = torch.exp(-cat_sigma).unsqueeze(0).repeat(samples, 1)
    alpha_s = torch.exp(-cat_sigma_next).unsqueeze(0).repeat(samples, 1)
    categorical_next, _ = diffusion._mdlm_update(logits, categorical_hat, alpha_t, alpha_s)
    if mode == "rollback" and changed_numeric.any():
        mask = torch.as_tensor(changed_numeric, device=device)
        noise = torch.randn(numeric_next.shape, device=device, dtype=numeric_next.dtype,
                            generator=rollback_generator)
        rollback_state = denoised + sigma_next.unsqueeze(0) * noise
        numeric_next[mask] = rollback_state[mask]
    for column, pos in layer.cat_pos.items():
        rows = np.flatnonzero(changed_cat[:, pos])
        if not len(rows): continue
        row_tensor = torch.as_tensor(rows, device=device)
        if mode == "rollback":
            categorical_next[row_tensor, pos] = int(diffusion.num_classes[pos])
        else:
            values = [layer.encode_category(column, corrected.iloc[row][column]) for row in rows]
            categorical_next[row_tensor, pos] = torch.as_tensor(values, device=device)
    return numeric_next.detach(), categorical_next.detach()


def build_action(frame, previous_frame, edge_list, layer, target_t, cutoffs, mode,
                 min_support, drift):
    samples = len(frame); corrected = frame.copy()
    changed_numeric = np.zeros((samples, len(layer.numeric_names)), dtype=bool)
    changed_cat = np.zeros((samples, len(layer.categorical_names)), dtype=bool)
    claimed = [set() for _ in range(samples)]; choices = Counter(); by_edge = Counter()
    ever = np.zeros(samples, dtype=bool)
    current_numeric = frame[layer.numeric_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    previous_numeric = previous_frame[layer.numeric_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    for edge in edge_list:
        if edge.get("usage_role", "actuate") != "actuate": continue
        edge_id = edge["candidate_id"]
        if target_t > cutoffs.get(edge_id, -1): continue
        active, _, violation = edge_masks(frame, edge)
        previous_active = edge_masks(previous_frame, edge)[0]
        stable = np.ones(samples, dtype=bool)
        columns = {atom["left_column"] for atom in edge.get("applicability", []) + edge.get("requirements", [])}
        columns |= {atom["right_column"] for atom in edge.get("requirements", []) if atom.get("right_kind") == "column"}
        for column in columns:
            if column in layer.num_pos:
                stable &= (np.abs(current_numeric[:, layer.num_pos[column]] -
                                  previous_numeric[:, layer.num_pos[column]]) /
                           layer.scales[column]) <= drift
        gate = active & violation & previous_active & stable
        if int(active.sum()) < min_support: gate[:] = False
        gate = calibrated_gate(frame, edge, gate, layer)
        for row_index in np.flatnonzero(gate):
            original = corrected.iloc[[row_index]].copy()
            action_mode = "reward" if mode in {"reward", "rollback"} else "directed"
            label, candidate, action_columns, _ = layer.choose(original, edge, action_mode)
            if not action_columns or claimed[row_index].intersection(action_columns): continue
            claimed[row_index].update(action_columns)
            if mode != "rollback":
                corrected.loc[corrected.index[row_index], action_columns] = candidate.iloc[0][action_columns].to_numpy()
            for column in action_columns:
                if column in layer.num_pos: changed_numeric[row_index, layer.num_pos[column]] = True
                elif column in layer.cat_pos: changed_cat[row_index, layer.cat_pos[column]] = True
            choices[label] += 1; by_edge[edge_id] += 1; ever[row_index] = True
    return corrected, changed_numeric, changed_cat, {
        "action_events": int(sum(by_edge.values())), "unique_rows_actuated": int(ever.sum()),
        "acted_rows": np.flatnonzero(ever).astype(int).tolist(),
        "actions_by_edge": dict(by_edge), "action_choices": dict(choices),
    }


@torch.no_grad()
def continue_unguided(diffusion, preview, layer, numeric, categorical, last_index, times, sn, sc, snx, scx):
    for index in reversed(range(last_index + 1)):
        numeric_hat, categorical_hat, denoised, raw_logits, _ = preview_step(
            diffusion, preview, numeric, categorical, index, times, sn, sc)
        empty_num = np.zeros((len(numeric), len(numeric[0])), dtype=bool)
        empty_cat = np.zeros((len(categorical), len(categorical[0])), dtype=bool)
        numeric, categorical = apply_update(
            diffusion, layer, numeric_hat, categorical_hat, denoised, raw_logits, None,
            empty_num, empty_cat, sn[index], snx[index], sc[index], scx[index], "base", None)
    return numeric, categorical


@torch.no_grad()
def branch_sample(diffusion, preview, train_data, info, edges, layer, cutoffs, samples,
                  target_t, min_support, drift):
    started = time.perf_counter(); device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sn = diffusion.num_schedule.total_noise(times); sc = diffusion.cat_schedule.total_noise(times)
    snx = torch.zeros_like(sn); snx[1:] = sn[:-1]
    scx = torch.zeros_like(sc); scx[1:] = sc[:-1]
    target_index = int(torch.argmin((times.squeeze() - target_t).abs()).item())
    numeric = torch.randn((samples, diffusion.num_numerical_features), device=device) * sn[-1]
    categorical = diffusion._sample_masked_prior(samples, len(diffusion.num_classes))
    previous_frame = None
    for index in reversed(range(target_index + 1, diffusion.num_timesteps)):
        numeric_hat, categorical_hat, denoised, raw_logits, codes = preview_step(
            diffusion, preview, numeric, categorical, index, times, sn, sc)
        previous_frame = decode(train_data, torch.cat([denoised, codes.float()], dim=1), info)
        empty_num = np.zeros((samples, len(layer.numeric_names)), dtype=bool)
        empty_cat = np.zeros((samples, len(layer.categorical_names)), dtype=bool)
        numeric, categorical = apply_update(
            diffusion, layer, numeric_hat, categorical_hat, denoised, raw_logits, previous_frame,
            empty_num, empty_cat, sn[index], snx[index], sc[index], scx[index], "base", None)

    numeric_hat, categorical_hat, denoised, raw_logits, codes = preview_step(
        diffusion, preview, numeric, categorical, target_index, times, sn, sc)
    frame = decode(train_data, torch.cat([denoised, codes.float()], dim=1), info)
    if previous_frame is None: previous_frame = frame.copy()
    branch_rng = get_rng_state(); rollback_generator = torch.Generator(device=device)
    rollback_generator.manual_seed(2026100611)
    branch_states = []; branch_stats = {}; post_step_rng = None
    for branch_index, mode in enumerate(["base", "directed", "reward", "rollback"]):
        if mode == "base":
            corrected = frame.copy()
            changed_numeric = np.zeros((samples, len(layer.numeric_names)), dtype=bool)
            changed_cat = np.zeros((samples, len(layer.categorical_names)), dtype=bool)
            stats = {"action_events": 0, "unique_rows_actuated": 0, "acted_rows": [],
                     "actions_by_edge": {}, "action_choices": {}}
        else:
            corrected, changed_numeric, changed_cat, stats = build_action(
                frame, previous_frame, edges, layer, float(times[target_index]), cutoffs,
                mode, min_support, drift)
        set_rng_state(branch_rng)
        next_numeric, next_categorical = apply_update(
            diffusion, layer, numeric_hat.clone(), categorical_hat.clone(), denoised.clone(), raw_logits.clone(),
            corrected, changed_numeric, changed_cat, sn[target_index], snx[target_index],
            sc[target_index], scx[target_index], mode, rollback_generator)
        if branch_index == 0: post_step_rng = get_rng_state()
        set_rng_state(post_step_rng)
        final_numeric, final_categorical = continue_unguided(
            diffusion, preview, layer, next_numeric, next_categorical, target_index - 1,
            times, sn, sc, snx, scx)
        branch_states.append(torch.cat([final_numeric, final_categorical.float()], dim=1).cpu())
        branch_stats[mode] = stats
    branch_stats["runtime_seconds"] = time.perf_counter() - started
    branch_stats["target_t"] = float(times[target_index]); branch_stats["target_index"] = target_index
    return branch_states, branch_stats


def selection_scores(frames, base, edges, layer, edit_weight=.5):
    samples = len(base); allowed = {}
    for edge in edges:
        active, _, bad = edge_masks(base, edge)
        legitimate = np.zeros(samples, dtype=bool)
        if edge["classification"] != "hard":
            keep = int(round(float(edge["calibration"]["violation_rate"]) * int(active.sum())))
            indices = np.flatnonzero(bad)
            if keep:
                severity = layer.severity(base, edge)[indices]
                legitimate[indices[np.argsort(severity)[:keep]]] = True
        allowed[edge["candidate_id"]] = legitimate
    all_scores = []
    base_num = base[layer.numeric_names].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    for branch_index, frame in enumerate(frames):
        score = np.zeros(samples, dtype=float)
        for edge in edges:
            weight = float(edge.get("selection_weight", 100.0))
            if weight <= 0: continue
            _, _, bad = edge_masks(frame, edge)
            score += weight * (bad & ~allowed[edge["candidate_id"]])
        numeric = frame[layer.numeric_names].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        density = layer.knn.kneighbors(layer.scaler.transform(numeric), return_distance=True)[0][:, -1] / layer.density_scale
        edit = np.zeros(samples, dtype=float)
        for pos, column in enumerate(layer.numeric_names):
            edit += np.abs(numeric[:, pos] - base_num[:, pos]) / layer.scales[column]
        for column in layer.categorical_names:
            edit += frame[column].astype(str).to_numpy() != base[column].astype(str).to_numpy()
        score += density + edit_weight * edit + (0.02 if branch_index else 0.0)
        all_scores.append(score)
    return np.column_stack(all_scores)


def select_rows(frames, scores):
    chosen = scores.argmin(axis=1); result = frames[0].copy()
    for branch_index in range(1, len(frames)):
        rows = np.flatnonzero(chosen == branch_index)
        if len(rows): result.iloc[rows] = frames[branch_index].iloc[rows].to_numpy()
    return result, chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096); parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261006); parser.add_argument("--target-t", type=float, default=.18)
    parser.add_argument("--min-support", type=int, default=12); parser.add_argument("--drift", type=float, default=.10)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    seed_all(args.seed); sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP
    info = json.loads((args.repo / "data/shoppers/info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, "shoppers")
    train, heldout = decode(train_data, train_data.X, info), decode(train_data, test_data.X, info)
    edges = json.loads(args.graph.read_text()); schedule = pd.read_csv(args.schedule)
    cutoffs = {str(row.candidate_id): float(row.cutoff) for row in schedule.itertuples(index=False)
               if bool(row.readable) and np.isfinite(row.cutoff)}
    edges = [edge for edge in edges if edge["candidate_id"] in cutoffs]
    with args.config.open("rb") as handle: config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"]); params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["denoise_fn"], strict=True); preview = PreviewModel(model, train_data.categories)
    dp = copy.deepcopy(config["diffusion_params"]); dp.update(num_timesteps=args.steps,
        scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    dp["sampler_params"] = copy.deepcopy(dp["sampler_params"])
    dp["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, preview, None,
        **dp, device=torch.device("cuda")).cuda()
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"]); diffusion.eval()
    layer = RowActionLayer(edges, train_data, train, info, 1.0, .5); common = args.seed + 100000

    seed_all(common); official_encoded = diffusion.sample(args.samples)
    official = decode(train_data, official_encoded, info)
    seed_all(common); encoded_branches, branch_stats = branch_sample(
        diffusion, preview, train_data, info, edges, layer, cutoffs, args.samples,
        args.target_t, args.min_support, args.drift)
    names = ["branch0_no_action", "branch_directed", "branch_reward", "branch_rollback"]
    frames = [decode(train_data, encoded, info) for encoded in encoded_branches]
    rows = [evaluate(official, official, heldout, edges, info, "official_unconditional", {}, layer)]
    edge_rows = edge_metric_rows(official, heldout, edges, "official_unconditional")
    for name, frame, mode in zip(names, frames, ["base", "directed", "reward", "rollback"]):
        stats = branch_stats[mode].copy(); stats["runtime_seconds"] = branch_stats["runtime_seconds"]
        rows.append(evaluate(frame, official, heldout, edges, info, name, stats, layer))
        edge_rows += edge_metric_rows(frame, heldout, edges, name)
    scores = selection_scores(frames, frames[0], edges, layer)
    selected, chosen = select_rows(frames, scores)
    selected_rows = np.flatnonzero(chosen != 0).astype(int).tolist()
    selected_stats = {"action_events": len(selected_rows), "unique_rows_actuated": len(selected_rows),
                      "acted_rows": selected_rows, "runtime_seconds": branch_stats["runtime_seconds"]}
    rows.append(evaluate(selected, official, heldout, edges, info, "mid_lookahead_select", selected_stats, layer))
    edge_rows += edge_metric_rows(selected, heldout, edges, "mid_lookahead_select")
    # A two-pass best-of baseline is slightly more expensive than the 1.6-pass
    # late lookahead and therefore a conservative compute-matched comparator.
    seed_all(common + 1); independent = decode(train_data, diffusion.sample(args.samples), info)
    best_scores = selection_scores([official, independent], official, edges, layer, edit_weight=0.0)
    best_of_two, best_chosen = select_rows([official, independent], best_scores)
    best_rows = np.flatnonzero(best_chosen != 0).astype(int).tolist()
    best_stats = {"action_events": len(best_rows), "unique_rows_actuated": len(best_rows),
                  "acted_rows": best_rows, "runtime_seconds": 0}
    rows.append(evaluate(best_of_two, official, heldout, edges, info, "best_of_2_full_select", best_stats, layer))
    edge_rows += edge_metric_rows(best_of_two, heldout, edges, "best_of_2_full_select")
    final_frame, final_stats = final_action(official, edges, layer, "reward")
    rows.append(evaluate(final_frame, official, heldout, edges, info, "final_reward_calibrated", final_stats, layer))
    edge_rows += edge_metric_rows(final_frame, heldout, edges, "final_reward_calibrated")
    sample_dir = args.output / "samples"; sample_dir.mkdir()
    sample_frames = {"official_unconditional": official, **dict(zip(names, frames)),
                     "mid_lookahead_select": selected, "best_of_2_full_select": best_of_two,
                     "final_reward_calibrated": final_frame}
    for name, sample_frame in sample_frames.items():
        sample_frame.to_csv(sample_dir / f"{name}.csv", index=False)
    metrics = pd.DataFrame(rows); metrics.to_csv(args.output / "method_metrics.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(args.output / "edge_metrics.csv", index=False)
    parity = bool(official.astype(str).equals(frames[0].astype(str)))
    diagnostics = {"parity_exact": parity, "target_t": branch_stats["target_t"],
        "target_index": branch_stats["target_index"], "branch_action_stats": branch_stats,
        "selected_branch_counts": {names[i]: int((chosen == i).sum()) for i in range(len(names))},
        "best_of_2_counts": {"official": int((best_chosen == 0).sum()),
                              "independent": int((best_chosen == 1).sum())},
        "equivalent_denoiser_passes": 1 + 3 * (branch_stats["target_index"] + 1) / args.steps}
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    lines = ["# Mid-trajectory branch lookahead", "", f"Exact no-action parity: `{parity}`. Target t: `{branch_stats['target_t']:.3f}`. Equivalent denoiser passes: `{diagnostics['equivalent_denoiser_passes']:.2f}`.", "",
        "| Method | Violation | Heldout error | Calibration error | KS | Drift | Density delta | Actions | Rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.method} | {row.macro_violation:.4f} | {row.heldout_violation_error:.4f} | {row.calibration_violation_error:.4f} | {row.numeric_ks:.4f} | {row.numeric_drift_sd:.4f} | {row.affected_density_delta:.4f} | {int(row.action_events)} | {int(row.unique_rows_actuated)} |")
    lines += ["", f"Selected branches: `{diagnostics['selected_branch_counts']}`",
              f"Best-of-2 choices: `{diagnostics['best_of_2_counts']}`"]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
