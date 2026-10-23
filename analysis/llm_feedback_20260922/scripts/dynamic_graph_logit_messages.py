#!/usr/bin/env python3
"""Trajectory-dependent probabilistic graph messages for TabDiff.

The method is a tabular mechanism study inspired by direct clean-logit
correction in GILC and loss/reward guidance.  It is not a reproduction of
those papers.  Instead of deciding graph applicability from an argmax row,
each implication sends a confidence-weighted message from its antecedent to
the consequent categorical logits.  Multiple passes propagate information
through a connected component within the same reverse step.

Soft relations target their train-only calibrated satisfaction probability;
an optional dual controller increases guidance only while predicted violation
mass exceeds that target.  An optional selective re-mask gives absorbing
categorical diffusion a bounded way to revisit an already committed error.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditioned_tabdiff_adapter import decode, seed_all
from harpoon_graph_proxy import categorical_code, categorical_layout, target_spec
from knowledge_graph_protocol import predicate_mask
from subgroup_projection_online import PreviewModel
from tabdiff_probe import preprocess_dataset
from typed_dynamic_graph_guidance import edge_masks, graph_metrics
from dynamic_cdd_graph_projection import calibrated_final_repair


def atom_probability(atom: dict, frame: pd.DataFrame, categorical: torch.Tensor,
                     logits: torch.Tensor, train_data, positions: dict[str, int],
                     sizes: list[int], offsets: list[int]) -> torch.Tensor:
    """Probability that an atom is true under the current clean prediction."""
    column = atom["left_column"]
    if (column in positions and atom.get("right_kind") == "literal"
            and atom.get("operator") in {"==", "!="}):
        position = positions[column]; size = sizes[position]
        target = categorical_code(train_data, position, atom["right_value"])
        probability = torch.softmax(logits[:, offsets[position]:offsets[position] + size], dim=-1)[:, target]
        current = categorical[:, position]
        fixed = current.lt(size)
        probability = torch.where(fixed, current.eq(target).to(probability.dtype), probability)
        return probability if atom["operator"] == "==" else 1.0 - probability
    mask = predicate_mask(frame, atom).astype(np.float32)
    return torch.as_tensor(mask, device=logits.device, dtype=logits.dtype)


def active_probability(edge: dict, frame: pd.DataFrame, categorical: torch.Tensor,
                       logits: torch.Tensor, train_data, positions: dict[str, int],
                       sizes: list[int], offsets: list[int]) -> torch.Tensor:
    result = torch.ones(len(frame), device=logits.device, dtype=logits.dtype)
    for atom in edge.get("applicability", []):
        result *= atom_probability(atom, frame, categorical, logits, train_data, positions, sizes, offsets)
    # Exception groups are an OR of conjunctions.  Use the exact product form
    # under the factorized clean-prediction approximation.
    no_exception = torch.ones_like(result)
    for group in edge.get("exceptions", []):
        group_probability = torch.ones_like(result)
        for atom in group.get("all_of", []):
            group_probability *= atom_probability(atom, frame, categorical, logits, train_data, positions, sizes, offsets)
        no_exception *= 1.0 - group_probability
    return result * no_exception


def certainty_gate(active: torch.Tensor, specification: dict) -> torch.Tensor:
    """Suppress uncertain edges without imposing a globally fixed start step."""
    threshold = specification.get("gate_tau")
    if threshold is None:
        return active
    temperature = float(specification.get("gate_temperature", 0.08))
    return active * torch.sigmoid((active - float(threshold)) / temperature)


def time_weight(t: float, start_t: float, schedule: str) -> float:
    if t > start_t:
        return 0.0
    progress = float(np.clip((start_t - t) / max(start_t, 1e-8), 0.0, 1.0))
    if schedule == "flat":
        return 1.0
    if schedule == "late_ramp":
        return 0.25 + 0.75 * progress
    if schedule == "middle_peak":
        return 0.25 + 3.0 * (4.0 * progress * (1.0 - progress))
    raise ValueError(schedule)


def method_specs(eta: float, dual_lr: float) -> list[dict]:
    return [
        {"name": "fixed_message_p1", "passes": 1, "dual": False, "kind": "linear", "schedule": "flat", "remask": False,
         "eta": eta, "dual_lr": dual_lr, "propagate": True},
        {"name": "fixed_message_p3", "passes": 3, "dual": False, "kind": "linear", "schedule": "flat", "remask": False,
         "eta": eta, "dual_lr": dual_lr, "propagate": True},
        # Matched-budget controls distinguish graph propagation from simply
        # applying a larger amount of logit guidance.
        {"name": "fixed_message_p1_budget3", "passes": 1, "dual": False, "kind": "linear", "schedule": "flat", "remask": False,
         "eta": 3.0 * eta, "dual_lr": dual_lr, "propagate": True},
        {"name": "fixed_message_p3_normalized", "passes": 3, "dual": False, "kind": "linear", "schedule": "flat", "remask": False,
         "eta": eta / 3.0, "dual_lr": dual_lr, "propagate": True},
        # Uses the same three correction passes but freezes antecedent routing
        # to the uncorrected logits, disabling within-step graph propagation.
        {"name": "fixed_message_p3_frozen_routing", "passes": 3, "dual": False, "kind": "linear", "schedule": "flat", "remask": False,
         "eta": eta, "dual_lr": dual_lr, "propagate": False},
        {"name": "certainty_message_p3_tau50", "passes": 3, "dual": False, "kind": "linear", "schedule": "flat", "remask": False,
         "eta": eta, "dual_lr": dual_lr, "propagate": True, "gate_tau": 0.50, "gate_temperature": 0.08},
        {"name": "certainty_message_p3_tau75", "passes": 3, "dual": False, "kind": "linear", "schedule": "flat", "remask": False,
         "eta": eta, "dual_lr": dual_lr, "propagate": True, "gate_tau": 0.75, "gate_temperature": 0.08},
        {"name": "dual_message_p1", "passes": 1, "dual": True, "kind": "linear", "schedule": "late_ramp", "remask": False,
         "eta": eta, "dual_lr": dual_lr, "propagate": True},
        {"name": "dual_message_p3", "passes": 3, "dual": True, "kind": "linear", "schedule": "late_ramp", "remask": False,
         "eta": eta, "dual_lr": dual_lr, "propagate": True},
        {"name": "dual_message_p3_middle", "passes": 3, "dual": True, "kind": "linear", "schedule": "middle_peak", "remask": False,
         "eta": eta, "dual_lr": dual_lr, "propagate": True},
        {"name": "dual_message_p3_remask", "passes": 3, "dual": True, "kind": "linear", "schedule": "late_ramp", "remask": True,
         "eta": eta, "dual_lr": dual_lr, "propagate": True},
        {"name": "sumproduct_message_p1", "passes": 1, "dual": False, "kind": "sum_product", "schedule": "flat", "remask": False,
         "eta": eta / 4.0, "dual_lr": dual_lr, "propagate": True},
        {"name": "sumproduct_message_p3", "passes": 3, "dual": False, "kind": "sum_product", "schedule": "flat", "remask": False,
         "eta": eta / 4.0, "dual_lr": dual_lr, "propagate": True},
    ]


@torch.no_grad()
def sample_messages(diffusion, preview: PreviewModel, train_data, info: dict,
                    edges: list[dict], samples: int, specification: dict,
                    start_t: float) -> tuple[torch.Tensor, dict]:
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num = diffusion.num_schedule.total_noise(times)
    sigma_cat = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num); sigma_num_next[1:] = sigma_num[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat); sigma_cat_next[1:] = sigma_cat[:-1]
    if diffusion.sampler_params["stochastic_sampler"] or diffusion.sampler_params["second_order_correction"]:
        raise ValueError("message sampler expects deterministic first-order numerical sampling")

    _, positions, sizes = categorical_layout(train_data, info)
    offsets = [int(sum(size + 1 for size in sizes[:position])) for position in range(len(sizes))]
    targets = {edge["candidate_id"]: target_spec(edge, train_data, positions, sizes) for edge in edges}
    desired = {edge["candidate_id"]: 1.0 - float(edge["calibration"]["violation_rate"]) for edge in edges}
    dual = {edge["candidate_id"]: 0.0 for edge in edges}
    numeric = torch.randn((samples, diffusion.num_numerical_features), device=device) * sigma_num[-1]
    categorical = diffusion._sample_masked_prior(samples, len(diffusion.num_classes))
    trace = []; total_messages = total_reopened = 0
    message_by_edge = {edge["candidate_id"]: 0 for edge in edges}
    reopen_by_edge = {edge["candidate_id"]: 0 for edge in edges}

    for index in reversed(range(diffusion.num_timesteps)):
        t = float(times[index]); weight = time_weight(t, start_t, specification["schedule"])
        # Preserve exact base-sampler RNG advancement.
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
        corrected = raw_logits.clone()
        step = {"t": t, "weight": weight, "messages": 0, "reopened": 0, "edges": {}}

        if weight > 0:
            # Update duals once from the unmodified current prediction.  The
            # correction passes below then propagate messages on this fixed
            # timestep without pretending they are independent observations.
            for edge in edges:
                edge_id = edge["candidate_id"]
                position, target, offset = targets[edge_id]
                active = certainty_gate(
                    active_probability(edge, frame, categorical_hat, corrected,
                                       train_data, positions, sizes, offsets),
                    specification,
                )
                target_probability = torch.softmax(
                    corrected[:, offset:offset + sizes[position]], dim=-1)[:, target]
                mass = active.sum().clamp_min(1e-8)
                predicted_violation = float((active * (1.0 - target_probability)).sum().div(mass).cpu())
                decoded_active, _, decoded_bad = edge_masks(frame, edge)
                decoded_violation = (float(decoded_bad.sum() / decoded_active.sum())
                                     if decoded_active.sum() else 0.0)
                # Probability mass can look calibrated while its argmax rows
                # remain invalid.  The executable verifier therefore controls
                # the dual, while soft probabilities route its messages.
                error = decoded_violation - float(edge["calibration"]["violation_rate"])
                if specification["dual"]:
                    dual[edge_id] = max(0.0, 0.9 * dual[edge_id] + specification["dual_lr"] * error)
                step["edges"][edge_id] = {"predicted_violation": predicted_violation,
                                           "decoded_violation": decoded_violation,
                                           "dual": dual[edge_id], "active_mass": float(mass.cpu())}

            for _ in range(specification["passes"]):
                for edge in edges:
                    edge_id = edge["candidate_id"]
                    position, target, offset = targets[edge_id]
                    routing_logits = corrected if specification["propagate"] else raw_logits
                    active = certainty_gate(
                        active_probability(edge, frame, categorical_hat, routing_logits,
                                           train_data, positions, sizes, offsets),
                        specification,
                    )
                    q = torch.softmax(corrected[:, offset:offset + sizes[position]], dim=-1)
                    target_probability = q[:, target]
                    if specification["kind"] == "sum_product":
                        # Directed sum-product factor for A -> B.  No backward
                        # message is sent to A, preventing antecedent evasion.
                        confidence = desired[edge_id]
                        numerator = 1.0 - active + active * confidence
                        denominator = 1.0 - active + active * (1.0 - confidence)
                        delta = specification["eta"] * weight * torch.log(
                            numerator.clamp_min(1e-8) / denominator.clamp_min(1e-8))
                    else:
                        gap = (desired[edge_id] - target_probability).clamp_min(0.0)
                        base_scale = dual[edge_id] if specification["dual"] else desired[edge_id]
                        delta = specification["eta"] * weight * base_scale * active * gap
                    editable = categorical_hat[:, position].eq(sizes[position])
                    corrected[editable, offset + target] += delta[editable]
                    count = int((editable & delta.gt(1e-6)).sum().cpu())
                    total_messages += count; step["messages"] += count; message_by_edge[edge_id] += count

            if specification["remask"]:
                # Re-open only excess decoded violations.  Ranking is based on
                # the soft message need, while the allowed exception mass is
                # taken exclusively from train calibration.
                claimed: set[int] = set()
                for edge in edges:
                    edge_id = edge["candidate_id"]
                    position, target, offset = targets[edge_id]
                    if position in claimed:
                        continue
                    active_mask, _, bad_mask = edge_masks(frame, edge)
                    bad_indices = np.flatnonzero(bad_mask)
                    allowed = int(round(float(edge["calibration"]["violation_rate"]) * int(active_mask.sum())))
                    excess = max(0, len(bad_indices) - allowed)
                    if excess <= 0:
                        continue
                    active = certainty_gate(
                        active_probability(edge, frame, categorical_hat, corrected,
                                           train_data, positions, sizes, offsets),
                        specification,
                    )
                    q = torch.softmax(corrected[:, offset:offset + sizes[position]], dim=-1)[:, target]
                    need = (active * (desired[edge_id] - q).clamp_min(0.0)).cpu().numpy()
                    selected = bad_indices[np.argsort(-need[bad_indices])[:excess]]
                    selected_tensor = torch.as_tensor(selected, device=device, dtype=torch.long)
                    wrong_fixed = categorical_hat[selected_tensor, position].lt(sizes[position]) & categorical_hat[selected_tensor, position].ne(target)
                    reopen = selected_tensor[wrong_fixed]
                    if len(reopen):
                        categorical_hat[reopen, position] = sizes[position]
                        base_scale = dual[edge_id] if specification["dual"] else desired[edge_id]
                        boost = specification["eta"] * weight * base_scale * torch.as_tensor(
                            need[selected[wrong_fixed.cpu().numpy()]], device=device, dtype=corrected.dtype)
                        corrected[reopen, offset + target] += boost
                        total_reopened += len(reopen); step["reopened"] += len(reopen)
                        reopen_by_edge[edge_id] += len(reopen); claimed.add(position)

        logits = diffusion._subs_parameterization(corrected, categorical_hat)
        alpha_t = torch.exp(-sigma_cat[index]).unsqueeze(0).repeat(samples, 1)
        alpha_s = torch.exp(-sigma_cat_next[index]).unsqueeze(0).repeat(samples, 1)
        categorical_next, _ = diffusion._mdlm_update(logits, categorical_hat, alpha_t, alpha_s)
        numeric, categorical = numeric_next.detach(), categorical_next.detach()
        trace.append(step)

    if not torch.all(categorical < diffusion.mask_index):
        raise RuntimeError("categorical mask remained at final step")
    return torch.cat([numeric, categorical.float()], dim=1).cpu(), {
        **specification, "start_t": start_t, "message_events": int(total_messages),
        "reopened_coordinates": int(total_reopened), "message_by_edge": message_by_edge,
        "reopen_by_edge": reopen_by_edge, "final_dual": dual, "trace": trace,
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
    parser.add_argument("--eta", type=float, default=4.0)
    parser.add_argument("--dual-lr", type=float, default=2.0)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    started = time.perf_counter()
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, args.dataset)
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

    samples_dir = args.output / "samples"; samples_dir.mkdir()
    common_seed = args.seed + 100000
    seed_all(common_seed)
    unconditional = decode(train_data, diffusion.sample(args.samples), info)
    unconditional.to_csv(samples_dir / "unconditional.csv", index=False)
    rows = [{"method": "unconditional", **graph_metrics(unconditional, heldout, edges)}]
    repaired, repair_diagnostics = calibrated_final_repair(unconditional, edges, args.seed)
    repaired.to_csv(samples_dir / "calibrated_final_repair.csv", index=False)
    rows.append({"method": "calibrated_final_repair", **graph_metrics(repaired, heldout, edges),
                 "message_events": repair_diagnostics["action_events"],
                 "reopened_coordinates": 0, "runtime_seconds": 0.0})
    diagnostics = {}
    for specification in method_specs(args.eta, args.dual_lr):
        seed_all(common_seed); tick = time.perf_counter()
        encoded, trace = sample_messages(diffusion, preview, train_data, info, edges,
                                         args.samples, specification, args.start_t)
        frame = decode(train_data, encoded, info)
        frame.to_csv(samples_dir / f"{specification['name']}.csv", index=False)
        trace["runtime_seconds"] = time.perf_counter() - tick
        diagnostics[specification["name"]] = trace
        rows.append({"method": specification["name"], **graph_metrics(frame, heldout, edges),
                     "message_events": trace["message_events"],
                     "reopened_coordinates": trace["reopened_coordinates"],
                     "runtime_seconds": trace["runtime_seconds"]})
    metrics = pd.DataFrame(rows); metrics.to_csv(args.output / "mechanism_metrics.csv", index=False)
    (args.output / "traces.json").write_text(json.dumps(diagnostics, indent=2))
    metadata = {
        "label": "dynamic probabilistic graph messages; GILC/LGD-inspired mechanism study",
        "not_official_reproduction": True, "samples": args.samples, "steps": args.steps,
        "seed": args.seed, "eta": args.eta, "dual_lr": args.dual_lr,
        "graph_edges": len(edges), "start_t": args.start_t, "api_calls": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
