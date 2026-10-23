#!/usr/bin/env python3
"""Dynamic graph activation with a CDD-style categorical projection layer.

This is a mechanism study, not an official reproduction of Constrained
Discrete Diffusion (CDD).  CDD supplies the KL-nearest feasibility layer; the
row/edge active set is supplied by a state-dependent tabular constraint graph.
An edge must be applicable in two consecutive clean-state previews after its
train-only readability cutoff.  ``persistent`` policies latch that edge for
the row so that later denoising cannot evade it by changing the antecedent.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cdd_tabdiff_categorical_projection import kl_argmax_projection
from conditioned_tabdiff_adapter import decode, seed_all
from harpoon_graph_proxy import categorical_layout, target_spec
from subgroup_projection_online import PreviewModel
from tabdiff_probe import preprocess_dataset
from typed_dynamic_graph_guidance import edge_masks, graph_metrics


def kl_cost(q: torch.Tensor, target: int) -> tuple[torch.Tensor, torch.Tensor]:
    p = kl_argmax_projection(q.clamp_min(1e-12), target).clamp_min(1e-12)
    return p, (q * (q.clamp_min(1e-12).log() - p.log())).sum(-1)


def choose_excess(indices: np.ndarray, costs: np.ndarray, budget: int, policy: str) -> np.ndarray:
    if budget <= 0:
        return np.empty(0, dtype=int)
    if budget >= len(indices):
        return indices
    # Minimum-KL is the fidelity-preserving action.  Severity is an ablation
    # that spends the same action budget on the most expensive violations.
    order = np.argsort(costs if policy == "calibrated_min_kl" else -costs)
    return indices[order[:budget]]


def calibrated_final_repair(frame: pd.DataFrame, edges: list[dict], seed: int) -> tuple[pd.DataFrame, dict]:
    """Post-hoc baseline that preserves each train-calibrated exception rate.

    The older generic repair utility only calibrated edges labelled near-hard;
    this graph intentionally contains soft edges, so calibration must apply to
    every edge.  With equal-cost categorical edits, rows are sampled using a
    fixed seed rather than selected with held-out information.
    """
    result = frame.copy(); rng = np.random.default_rng(seed)
    changed = np.zeros(len(result), dtype=bool); events = 0
    for edge in edges:
        active, _, bad = edge_masks(result, edge)
        indices = np.flatnonzero(bad)
        allowed = int(round(float(edge["calibration"]["violation_rate"]) * int(active.sum())))
        excess = max(0, len(indices) - allowed)
        if excess <= 0:
            continue
        selected = rng.choice(indices, size=excess, replace=False)
        requirement = edge["requirements"][0]
        if requirement.get("right_kind") != "literal" or requirement.get("operator") != "==":
            raise ValueError("pilot post-repair supports categorical literal equalities only")
        result.loc[selected, requirement["left_column"]] = requirement["right_value"]
        changed[selected] = True; events += len(selected)
    return result, {"action_events": int(events), "unique_rows_actuated": int(changed.sum())}


@torch.no_grad()
def sample_dynamic(diffusion, preview: PreviewModel, train_data, info: dict,
                   edges: list[dict], samples: int, mode: str,
                   start_t: float, stop_t: float) -> tuple[torch.Tensor, dict]:
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num = diffusion.num_schedule.total_noise(times)
    sigma_cat = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num); sigma_num_next[1:] = sigma_num[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat); sigma_cat_next[1:] = sigma_cat[:-1]
    if diffusion.sampler_params["stochastic_sampler"] or diffusion.sampler_params["second_order_correction"]:
        raise ValueError("dynamic CDD adapter expects deterministic first-order numeric sampling")

    _, positions, sizes = categorical_layout(train_data, info)
    specs = {edge["candidate_id"]: target_spec(edge, train_data, positions, sizes) for edge in edges}
    numeric = torch.randn((samples, diffusion.num_numerical_features), device=device) * sigma_num[-1]
    categorical = diffusion._sample_masked_prior(samples, len(diffusion.num_classes))
    previous_active = {edge["candidate_id"]: np.zeros(samples, dtype=bool) for edge in edges}
    latched = {edge["candidate_id"]: np.zeros(samples, dtype=bool) for edge in edges}
    total_selected = total_reopened = total_active = 0
    costs_all: list[float] = []
    trace: list[dict] = []
    selected_totals = {edge["candidate_id"]: 0 for edge in edges}
    reopened_totals = {edge["candidate_id"]: 0 for edge in edges}
    newly_latched_totals = {edge["candidate_id"]: 0 for edge in edges}

    for index in reversed(range(diffusion.num_timesteps)):
        t = float(times[index])
        # Match TabDiff's deterministic EDM update, including zero-amplitude
        # random draws that advance the RNG before categorical transitions.
        numeric_hat = numeric + (sigma_num[index].square() - sigma_num[index].square()).sqrt() * torch.randn_like(numeric)
        move_chance = -torch.expm1(sigma_cat[index] - sigma_cat[index])
        categorical_hat, _ = diffusion.q_xt(categorical, move_chance)
        categorical_oh = diffusion.to_one_hot(categorical_hat).to(numeric.dtype)
        denoised, raw_logits = preview.base(
            numeric_hat.float(), categorical_oh, times[index].squeeze().repeat(samples),
            sigma=sigma_num[index].unsqueeze(0).repeat(samples, 1),
        )
        direction = (numeric_hat - denoised) / sigma_num[index]
        numeric_next = numeric_hat + (sigma_num_next[index] - sigma_num[index]) * direction
        codes = preview.predicted_codes(categorical_oh, raw_logits)
        frame = decode(train_data, torch.cat([denoised, codes.float()], dim=1).cpu(), info)

        selected_by_edge: dict[str, np.ndarray] = {}
        step = {"t": t, "stable_active": 0, "latched": 0, "selected": 0, "reopened": 0}
        for edge in edges:
            edge_id = edge["candidate_id"]
            active, _, bad = edge_masks(frame, edge)
            stable_active = active & previous_active[edge_id]
            readable = t <= start_t
            if readable and mode.startswith("dynamic"):
                before = latched[edge_id].copy()
                latched[edge_id] |= stable_active
                newly_latched_totals[edge_id] += int((latched[edge_id] & ~before).sum())

            if mode == "static_full_hard":
                scope = active
            elif mode == "static_delayed_hard":
                scope = active if readable else np.zeros(samples, dtype=bool)
            elif mode == "dynamic_transient_hard":
                scope = stable_active if stop_t <= t <= start_t else np.zeros(samples, dtype=bool)
            elif mode.startswith("dynamic_persistent"):
                scope = latched[edge_id] if readable else np.zeros(samples, dtype=bool)
            else:
                raise ValueError(mode)

            candidates = np.flatnonzero(scope & bad)
            if len(candidates) and "calibrated" in mode:
                active_count = int(scope.sum())
                allowed = int(round(float(edge["calibration"]["violation_rate"]) * active_count))
                budget = max(0, len(candidates) - allowed)
                position, target, offset = specs[edge_id]
                q = torch.softmax(raw_logits[candidates, offset:offset + sizes[position]], dim=-1)
                _, costs = kl_cost(q, target)
                candidates = choose_excess(candidates, costs.cpu().numpy(), budget,
                                           "calibrated_min_kl" if mode.endswith("min_kl") else "calibrated_severity")
            selected_by_edge[edge_id] = candidates
            previous_active[edge_id] = active
            step["stable_active"] += int(stable_active.sum())
            step["latched"] += int(latched[edge_id].sum())

        # Re-mask a previously committed but now infeasible consequent.  This
        # is the discrete analogue of rollback before applying the feasibility
        # operator and is necessary because MDLM states otherwise cannot flip.
        for edge in edges:
            edge_id = edge["candidate_id"]
            rows = selected_by_edge[edge_id]
            if not len(rows):
                continue
            position, target, _ = specs[edge_id]
            row_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
            wrong = categorical_hat[row_tensor, position].ne(diffusion.mask_index[position]) & categorical_hat[row_tensor, position].ne(target)
            reopen = row_tensor[wrong]
            if len(reopen):
                categorical_hat[reopen, position] = diffusion.mask_index[position]
                total_reopened += len(reopen); step["reopened"] += len(reopen)
                reopened_totals[edge_id] += len(reopen)

        projected_logits = raw_logits.clone()
        for edge in edges:
            edge_id = edge["candidate_id"]
            rows = selected_by_edge[edge_id]
            if not len(rows):
                continue
            position, target, offset = specs[edge_id]
            row_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
            q = torch.softmax(raw_logits[row_tensor, offset:offset + sizes[position]], dim=-1).clamp_min(1e-12)
            p, costs = kl_cost(q, target)
            projected_logits[row_tensor, offset:offset + sizes[position]] = p.log()
            costs_all.extend(costs.cpu().tolist())
            total_selected += len(rows); step["selected"] += len(rows)
            selected_totals[edge_id] += len(rows)

        logits = diffusion._subs_parameterization(projected_logits, categorical_hat)
        alpha_t = torch.exp(-sigma_cat[index]).unsqueeze(0).repeat(samples, 1)
        alpha_s = torch.exp(-sigma_cat_next[index]).unsqueeze(0).repeat(samples, 1)
        categorical_next, q_xs = diffusion._mdlm_update(logits, categorical_hat, alpha_t, alpha_s)

        # CDD constrains the decoded categorical choice.  Preserve TabDiff's
        # learned mask/unmask time but greedily choose the feasible category
        # whenever a selected coordinate commits at this step.
        for edge in edges:
            edge_id = edge["candidate_id"]
            rows = selected_by_edge[edge_id]
            if not len(rows):
                continue
            position, target, _ = specs[edge_id]
            row_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
            masked = categorical_hat[row_tensor, position].eq(diffusion.mask_index[position])
            editable = row_tensor[masked]
            if len(editable):
                col_q = q_xs[editable, position]
                mask_idx = int(diffusion.mask_index[position])
                stay_probability = col_q[:, mask_idx] / col_q.sum(-1).clamp_min(1e-12)
                stay = torch.rand(len(editable), device=device) < stay_probability
                values = torch.full((len(editable),), target, device=device, dtype=categorical_next.dtype)
                values[stay] = mask_idx
                categorical_next[editable, position] = values

        total_active += step["latched"]
        trace.append(step)
        numeric, categorical = numeric_next.detach(), categorical_next.detach()

    if not torch.all(categorical < diffusion.mask_index):
        raise RuntimeError("categorical mask remained at final step")
    return torch.cat([numeric, categorical.float()], dim=1).cpu(), {
        "mode": mode, "start_t": start_t, "stop_t": stop_t,
        "selected_row_edge_steps": int(total_selected), "reopened_coordinates": int(total_reopened),
        "mean_projection_kl": float(np.mean(costs_all)) if costs_all else 0.0,
        "latched_row_edge_steps": int(total_active), "trace": trace,
        "selected_by_edge": selected_totals, "reopened_by_edge": reopened_totals,
        "unique_latched_by_edge": {key: int(value.sum()) for key, value in latched.items()},
        "newly_latched_events_by_edge": newly_latched_totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261376)
    parser.add_argument("--start-t", type=float, default=.50)
    parser.add_argument("--stop-t", type=float, default=.10)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    started = time.perf_counter()
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, args.dataset)
    train = decode(train_data, train_data.X, info)
    heldout = decode(train_data, test_data.X, info)
    edges = json.loads(args.graph.read_text())
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["denoise_fn"], strict=True)
    preview = PreviewModel(model, train_data.categories)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column",
                            cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, preview, None,
        **diffusion_params, device=torch.device("cuda")).cuda()
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()

    sample_dir = args.output / "samples"; sample_dir.mkdir()
    common_seed = args.seed + 100000
    seed_all(common_seed)
    unconditional = decode(train_data, diffusion.sample(args.samples), info)
    unconditional.to_csv(sample_dir / "unconditional.csv", index=False)

    modes = [
        "static_full_hard", "static_delayed_hard", "dynamic_transient_hard",
        "dynamic_persistent_hard", "dynamic_persistent_calibrated_min_kl",
        "dynamic_persistent_calibrated_severity",
    ]
    summary = [{"method": "unconditional", **graph_metrics(unconditional, heldout, edges)}]
    traces = {}
    for mode in modes:
        seed_all(common_seed)
        tick = time.perf_counter()
        encoded, diagnostics = sample_dynamic(diffusion, preview, train_data, info, edges,
                                              args.samples, mode, args.start_t, args.stop_t)
        frame = decode(train_data, encoded, info)
        frame.to_csv(sample_dir / f"{mode}.csv", index=False)
        diagnostics["runtime_seconds"] = time.perf_counter() - tick
        traces[mode] = diagnostics
        summary.append({"method": mode, **graph_metrics(frame, heldout, edges),
                        **{key: diagnostics[key] for key in ("selected_row_edge_steps", "reopened_coordinates", "mean_projection_kl", "runtime_seconds")}})

    repaired, repair_diag = calibrated_final_repair(unconditional, edges, args.seed)
    repaired.to_csv(sample_dir / "calibrated_final_repair.csv", index=False)
    summary.append({"method": "calibrated_final_repair", **graph_metrics(repaired, heldout, edges),
                    "selected_row_edge_steps": repair_diag["action_events"]})
    pd.DataFrame(summary).to_csv(args.output / "mechanism_metrics.csv", index=False)
    (args.output / "traces.json").write_text(json.dumps(traces, indent=2))
    metadata = {
        "label": "dynamic graph + CDD-style projection; not an official CDD reproduction",
        "source_method": "Constrained Discrete Diffusion, NeurIPS 2025",
        "samples": args.samples, "steps": args.steps, "seed": args.seed,
        "graph_edges": len(edges), "readability_start_t": args.start_t,
        "projection": "per-row exact D_KL(q||p) argmax-feasible I-projection",
        "activation": "two consecutive xhat0 applicability states; persistent variants latch row-edge",
        "api_calls": 0, "runtime_seconds": time.perf_counter() - started,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
