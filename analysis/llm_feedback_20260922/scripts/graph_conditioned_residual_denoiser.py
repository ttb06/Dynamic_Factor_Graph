#!/usr/bin/env python3
"""Train and sample a topology-aware graph-conditioned TabDiff adapter.

This is deliberately a new model family rather than another hand-tuned
inference-time correction.  A frozen TabDiff first predicts the clean record.
The adapter then (1) evaluates every typed graph edge softly, (2) propagates
edge-state embeddings over the constraint-graph topology, and (3) predicts
bounded residuals only for columns that occur as rule consequents.

The adapter is trained end-to-end with clean-record reconstruction, observed
edge applicability/satisfaction, train-only violation-rate calibration, and a
proximal term to the frozen backbone.  Consequently, empirical exceptions are
supervision rather than rows that are forcibly repaired.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from conditioned_tabdiff_adapter import decode
from finetune_mixed_graph_calibrated_tabdiff import (
    categorical_probabilities,
    checkpoint_state,
    compile_graph,
    select_trainable,
    soft_graph_state,
)
from tabdiff_probe import preprocess_dataset
from typed_dynamic_graph_guidance import TypedActuator, edge_masks, graph_metrics


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone(repo: Path, dataset: str, config_path: Path, checkpoint_path: Path,
                  device: torch.device, steps: int | None = None):
    sys.path.insert(0, str(repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    train_data, test_data = preprocess_dataset(repo, dataset)
    info = json.loads((repo / "data" / dataset / "info.json").read_text())
    with config_path.open("rb") as handle:
        config = copy.deepcopy(pickle.load(handle))
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["denoise_fn"], strict=True)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    if steps is not None:
        diffusion_params["num_timesteps"] = steps
        diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
        diffusion_params["sampler_params"].update(
            stochastic_sampler=False, second_order_correction=False
        )
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, model, None,
        **diffusion_params, device=device,
    ).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    return diffusion, train_data, test_data, info, config


def atom_columns(atom: dict) -> set[str]:
    result = {str(atom["left_column"])}
    if atom.get("right_kind") == "column":
        result.add(str(atom["right_column"]))
    return result


class GraphResidualAdapter(nn.Module):
    """Topology-aware residual controller over predicted-clean records."""

    def __init__(self, programs: list[dict], numeric_names: list[str], categorical_names: list[str],
                 num_classes: list[int], hidden: int = 128, graph_layers: int = 2,
                 numeric_scale: float = .35, logit_scale: float = 4.0,
                 truth_temperature: float = .20, window_min: float = .05,
                 window_max: float = .65, window_temperature: float = .03):
        super().__init__()
        if not programs:
            raise ValueError("adapter requires at least one compiled edge")
        self.programs = programs
        self.numeric_names = list(numeric_names)
        self.categorical_names = list(categorical_names)
        self.num_classes = [int(value) for value in num_classes]
        self.num_classes_w_mask = tuple(value + 1 for value in self.num_classes)
        self.truth_temperature = float(truth_temperature)
        self.numeric_scale = float(numeric_scale)
        self.logit_scale = float(logit_scale)
        self.window_min = float(window_min); self.window_max = float(window_max)
        self.window_temperature = float(window_temperature)
        self.edge_encoder = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.edge_identity = nn.Parameter(torch.zeros(len(programs), hidden))
        nn.init.normal_(self.edge_identity, std=.02)
        self.graph_linears = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(graph_layers))
        self.graph_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(graph_layers))

        source_sets, target_sets = [], []
        targets: dict[tuple[str, int], dict] = {}
        for edge_index, program in enumerate(programs):
            sources = {atom["left_column"] for atom in program.get("applicability", [])}
            source_sets.append(sources)
            edge_targets = set()
            for atom in program.get("requirements", []):
                column = atom["left_column"]
                if atom["kind"] == "numeric":
                    key = ("numeric", int(atom["left_index"]))
                else:
                    key = ("categorical", int(atom["left_index"]))
                edge_targets.add(column)
                targets.setdefault(key, {"column": column, "edges": []})["edges"].append(edge_index)
            target_sets.append(edge_targets)

        adjacency = torch.eye(len(programs), dtype=torch.float32)
        for left in range(len(programs)):
            for right in range(len(programs)):
                if (target_sets[left] & source_sets[right]) or (target_sets[right] & source_sets[left]) \
                        or (target_sets[left] & target_sets[right]):
                    adjacency[left, right] = 1.0
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adjacency", adjacency)

        self.target_specs = []
        self.heads = nn.ModuleDict()
        for (kind, index), payload in sorted(targets.items()):
            key = f"{kind}_{index}"
            local = 1 if kind == "numeric" else self.num_classes[index]
            output = 1 if kind == "numeric" else self.num_classes[index]
            head = nn.Sequential(
                nn.Linear(hidden + local + 1, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output),
            )
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
            self.heads[key] = head
            self.register_buffer(f"incidence_{key}", torch.as_tensor(payload["edges"], dtype=torch.long))
            self.target_specs.append({"kind": kind, "index": index, "key": key,
                                      "column": payload["column"], "edges": payload["edges"]})

        self.graph_view = SimpleNamespace(
            num_classes=np.asarray(self.num_classes),
            num_classes_w_mask=self.num_classes_w_mask,
        )

    def forward(self, predicted_num: torch.Tensor, raw_logits: torch.Tensor,
                t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        gates, satisfaction = soft_graph_state(
            predicted_num, raw_logits, self.graph_view, self.programs, self.truth_temperature
        )
        t = t.reshape(-1, 1).to(predicted_num.dtype)
        envelope = (torch.sigmoid((t - self.window_min) / self.window_temperature)
                    * torch.sigmoid((self.window_max - t) / self.window_temperature))
        target_rates = torch.as_tensor(
            [program["target_violation_rate"] for program in self.programs],
            dtype=predicted_num.dtype, device=predicted_num.device,
        ).reshape(1, -1).expand(len(predicted_num), -1)
        uncertainty = 4.0 * satisfaction * (1.0 - satisfaction)
        features = torch.stack([
            gates, satisfaction, gates * (1.0 - satisfaction), uncertainty,
            target_rates, t.expand(-1, len(self.programs)),
        ], dim=-1)
        hidden = self.edge_encoder(features) + self.edge_identity.unsqueeze(0)
        for linear, norm in zip(self.graph_linears, self.graph_norms):
            neighbourhood = torch.einsum("ef,bfh->beh", self.adjacency, hidden)
            hidden = norm(hidden + F.silu(linear(neighbourhood)))

        corrected_num = predicted_num.clone(); corrected_logits = raw_logits.clone()
        chunks = list(torch.split(raw_logits, self.num_classes_w_mask, dim=-1))
        offset = np.cumsum([0] + list(self.num_classes_w_mask)).tolist()
        residual_squared = predicted_num.sum() * 0.0
        for spec in self.target_specs:
            indices = getattr(self, f"incidence_{spec['key']}")
            context = hidden.index_select(1, indices).mean(dim=1)
            if spec["kind"] == "numeric":
                local = predicted_num[:, spec["index"]:spec["index"] + 1]
                raw_delta = self.heads[spec["key"]](torch.cat([context, local, t], dim=1))
                delta = envelope * self.numeric_scale * torch.tanh(raw_delta)
                corrected_num[:, spec["index"]] = local[:, 0] + delta[:, 0]
            else:
                position = spec["index"]; size = self.num_classes[position]
                local = torch.softmax(chunks[position][:, :size], dim=-1)
                raw_delta = self.heads[spec["key"]](torch.cat([context, local, t], dim=1))
                delta = envelope * self.logit_scale * torch.tanh(raw_delta)
                corrected_logits[:, offset[position]:offset[position] + size] += delta
            residual_squared = residual_squared + delta.square().mean()
        corrected_gates, corrected_satisfaction = soft_graph_state(
            corrected_num, corrected_logits, self.graph_view, self.programs, self.truth_temperature
        )
        diagnostics = {
            "base_gates": gates, "base_satisfaction": satisfaction,
            "gates": corrected_gates, "satisfaction": corrected_satisfaction,
            "residual_squared": residual_squared,
        }
        return corrected_num, corrected_logits, diagnostics


class GraphConditionedModel(nn.Module):
    """Two-pass denoiser: frozen TabDiff followed by the graph adapter."""

    def __init__(self, base: nn.Module, adapter: GraphResidualAdapter):
        super().__init__(); self.base = base; self.adapter = adapter

    def forward(self, x_num, x_cat, t, sigma=None):
        predicted_num, raw_logits = self.base(x_num, x_cat, t, sigma=sigma)
        corrected_num, corrected_logits, _ = self.adapter(predicted_num, raw_logits, t)
        return corrected_num, corrected_logits


def noisy_base_prediction(diffusion, batch: torch.Tensor, t_min: float, t_max: float):
    batch_size = len(batch); x_num = batch[:, :diffusion.num_numerical_features]
    x_cat = batch[:, diffusion.num_numerical_features:].long()
    t = t_min + (t_max - t_min) * torch.rand(batch_size, 1, device=batch.device, dtype=x_num.dtype)
    sigma_num = diffusion.num_schedule.total_noise(t); sigma_cat = diffusion.cat_schedule.total_noise(t)
    noisy_num = x_num + torch.randn_like(x_num) * sigma_num
    move_chance = -torch.expm1(-sigma_cat)
    if x_cat.shape[1]:
        strategy = "soft" if diffusion.cat_scheduler == "log_linear_per_column" else "hard"
        _, noisy_cat = diffusion.q_xt(x_cat, move_chance, strategy=strategy)
    else:
        noisy_cat = x_cat
    with torch.no_grad():
        predicted_num, raw_logits = diffusion._denoise_fn(
            noisy_num, noisy_cat, t.squeeze(-1), sigma=sigma_num
        )
    return predicted_num.detach(), raw_logits.detach(), t.squeeze(-1), x_num, x_cat


def unrolled_trajectory_prediction(diffusion, adapter: GraphResidualAdapter, batch: torch.Tensor,
                                   t_min: float, t_max: float, rollout_steps: int):
    """Differentiable expected reverse rollout from a forward-noised real row.

    Numeric states use TabDiff's deterministic Euler update.  Categorical
    states use the exact expected absorbing posterior: committed probability
    remains committed, while a fraction of the mask mass is allocated using
    the corrected clean distribution.  The frozen backbone remains
    differentiable with respect to its state input, so terminal losses train
    every adapter invocation in the truncated trajectory.
    """
    if rollout_steps < 2:
        return noisy_base_prediction(diffusion, batch, t_min, t_max)
    batch_size = len(batch); clean_num = batch[:, :diffusion.num_numerical_features]
    clean_cat = batch[:, diffusion.num_numerical_features:].long()
    start_scalar = t_min + (t_max - t_min) * torch.rand((), device=batch.device, dtype=clean_num.dtype)
    current_t = start_scalar.expand(batch_size, 1)
    sigma_num = diffusion.num_schedule.total_noise(current_t)
    sigma_cat = diffusion.cat_schedule.total_noise(current_t)
    numeric = clean_num + torch.randn_like(clean_num) * sigma_num
    move_chance = -torch.expm1(-sigma_cat)
    strategy = "soft" if diffusion.cat_scheduler == "log_linear_per_column" else "hard"
    _, categorical = diffusion.q_xt(clean_cat, move_chance, strategy=strategy)

    predicted_num = raw_logits = None
    for step in range(rollout_steps):
        fraction = step / max(rollout_steps - 1, 1)
        scalar_t = start_scalar * (1.0 - fraction) + torch.as_tensor(t_min, device=batch.device) * fraction
        current_t = scalar_t.expand(batch_size, 1)
        sigma_num = diffusion.num_schedule.total_noise(current_t)
        sigma_cat = diffusion.cat_schedule.total_noise(current_t)
        predicted_num, raw_logits = diffusion._denoise_fn(
            numeric, categorical, current_t.squeeze(-1), sigma=sigma_num
        )
        corrected_num, corrected_logits, _ = adapter(
            predicted_num, raw_logits, current_t.squeeze(-1)
        )
        if step == rollout_steps - 1:
            predicted_num, raw_logits = predicted_num, raw_logits
            break
        next_fraction = (step + 1) / (rollout_steps - 1)
        next_scalar = start_scalar * (1.0 - next_fraction) + torch.as_tensor(t_min, device=batch.device) * next_fraction
        next_t = next_scalar.expand(batch_size, 1)
        sigma_num_next = diffusion.num_schedule.total_noise(next_t)
        numeric = numeric + (sigma_num_next - sigma_num) * (numeric - corrected_num) / sigma_num.clamp_min(1e-6)

        sigma_cat_next = diffusion.cat_schedule.total_noise(next_t)
        alpha_t = torch.exp(-sigma_cat); alpha_next = torch.exp(-sigma_cat_next)
        reveal = ((alpha_next - alpha_t) / (1.0 - alpha_t).clamp_min(1e-6)).clamp(0.0, 1.0)
        state_chunks = torch.split(categorical, diffusion.num_classes_w_mask, dim=-1)
        logit_chunks = torch.split(corrected_logits, diffusion.num_classes_w_mask, dim=-1)
        next_chunks = []
        for position, (state, logits, size) in enumerate(zip(state_chunks, logit_chunks, diffusion.num_classes)):
            size = int(size); mask_mass = state[:, size:size + 1]
            probability = torch.softmax(logits[:, :size], dim=-1)
            amount = reveal[:, position:position + 1]
            next_nonmask = state[:, :size] + mask_mass * amount * probability
            next_mask = mask_mass * (1.0 - amount)
            next_chunks.append(torch.cat([next_nonmask, next_mask], dim=1))
        categorical = torch.cat(next_chunks, dim=1)
    assert predicted_num is not None and raw_logits is not None
    return predicted_num, raw_logits, current_t.squeeze(-1), clean_num, clean_cat


def adapter_loss(adapter: GraphResidualAdapter, diffusion, predicted_num, raw_logits, t,
                 clean_num, clean_cat, labels, reconstruction_weight: float,
                 applicability_weight: float, relation_weight: float, rate_weight: float,
                 proximity_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    corrected_num, corrected_logits, states = adapter(predicted_num, raw_logits, t)
    recon_terms = []
    for spec in adapter.target_specs:
        if spec["kind"] == "numeric":
            recon_terms.append(F.smooth_l1_loss(
                corrected_num[:, spec["index"]], clean_num[:, spec["index"]], beta=.1
            ))
        else:
            position = spec["index"]
            chunks = torch.split(corrected_logits, adapter.num_classes_w_mask, dim=-1)
            recon_terms.append(F.cross_entropy(
                chunks[position][:, :adapter.num_classes[position]], clean_cat[:, position]
            ))
    reconstruction = torch.stack(recon_terms).mean()
    clean_active = labels[:, :, 0]; clean_satisfied = labels[:, :, 1]
    applicability = F.binary_cross_entropy(states["gates"], clean_active)
    relation_terms = F.binary_cross_entropy(states["satisfaction"], clean_satisfied, reduction="none")
    relation = (relation_terms * clean_active).sum() / clean_active.sum().clamp_min(1.0)
    target_rates = torch.as_tensor(
        [program["target_violation_rate"] for program in adapter.programs],
        dtype=predicted_num.dtype, device=predicted_num.device,
    )
    predicted_rates = (states["gates"] * (1.0 - states["satisfaction"])).sum(0) \
        / states["gates"].sum(0).clamp_min(1e-5)
    rate = F.smooth_l1_loss(predicted_rates, target_rates, beta=.02)

    proximal_terms = []
    for spec in adapter.target_specs:
        if spec["kind"] == "numeric":
            proximal_terms.append(F.mse_loss(
                corrected_num[:, spec["index"]], predicted_num[:, spec["index"]]
            ))
        else:
            position = spec["index"]; size = adapter.num_classes[position]
            base_chunk = torch.split(raw_logits, adapter.num_classes_w_mask, dim=-1)[position][:, :size]
            corrected_chunk = torch.split(corrected_logits, adapter.num_classes_w_mask, dim=-1)[position][:, :size]
            proximal_terms.append(F.kl_div(
                F.log_softmax(corrected_chunk, dim=-1), F.softmax(base_chunk, dim=-1), reduction="batchmean"
            ))
    proximity = torch.stack(proximal_terms).mean()
    total = (reconstruction_weight * reconstruction + applicability_weight * applicability
             + relation_weight * relation + rate_weight * rate + proximity_weight * proximity)
    return total, {"reconstruction": reconstruction, "applicability": applicability,
                   "relation": relation, "rate": rate, "proximity": proximity,
                   "residual": states["residual_squared"],
                   "predicted_rate": predicted_rates.mean(), "target_rate": target_rates.mean()}


def prepare_graph(diffusion, train_data, info: dict, graph_path: Path, calibration_target: str,
                  min_active_rows: int):
    raw_frame = decode(train_data, train_data.X, info)
    edges = json.loads(graph_path.read_text())
    programs, labels, skipped = compile_graph(
        edges, raw_frame, train_data, info, min_active_rows, None, calibration_target
    )
    numeric_names = [info["column_names"][index] for index in info["num_col_idx"]]
    categorical_names = [info["column_names"][index]
                         for index in info["target_col_idx"] + info["cat_col_idx"]]
    return raw_frame, edges, programs, labels, skipped, numeric_names, categorical_names


def calibrated_minimal_final_repair(frame: pd.DataFrame, actuator: TypedActuator) -> tuple[pd.DataFrame, dict]:
    """Strong post-hoc baseline with the same per-edge empirical budget.

    Every edge, including soft edges, retains its train-calibrated violation
    mass.  Among excess violations, the cheapest raw-space edits are applied;
    applicability columns are protected whenever the consequent is a
    two-column inequality.
    """
    result = frame.copy(); changed = np.zeros(len(result), dtype=bool); events = 0
    by_edge = {}
    for rule in actuator.compiled:
        active, _, bad = edge_masks(result, rule)
        indices = np.flatnonzero(bad)
        target_rate = float(rule.get("calibration", {}).get("violation_rate", 0.0))
        allowed = int(round(target_rate * int(active.sum())))
        excess = max(0, len(indices) - allowed)
        if excess == 0: continue
        requirement = rule["requirement"]; kind = rule["kind"]
        if kind == "categorical_literal":
            costs = np.zeros(len(indices), dtype=float)
        elif kind == "numeric_literal":
            column = requirement["left_column"]
            values = pd.to_numeric(result.loc[indices, column], errors="coerce").to_numpy(float)
            costs = np.abs(values - float(rule["target_raw"])) / actuator.scales[column]
        else:
            left, right = requirement["left_column"], requirement["right_column"]
            difference = np.abs(pd.to_numeric(result.loc[indices, left], errors="coerce").to_numpy(float)
                                - pd.to_numeric(result.loc[indices, right], errors="coerce").to_numpy(float))
            costs = difference / max(actuator.scales[left], actuator.scales[right])
        selected = indices[np.argsort(costs, kind="stable")[:excess]]
        if kind == "categorical_literal":
            result.loc[selected, requirement["left_column"]] = requirement["right_value"]
        elif kind == "numeric_literal":
            result.loc[selected, requirement["left_column"]] = rule["target_raw"]
        else:
            left, right = requirement["left_column"], requirement["right_column"]
            protected = {atom["left_column"] for atom in rule.get("applicability", [])}
            if left in protected and right not in protected:
                result.loc[selected, right] = result.loc[selected, left].to_numpy()
            else:
                result.loc[selected, left] = result.loc[selected, right].to_numpy()
        changed[selected] = True; events += len(selected); by_edge[rule["candidate_id"]] = len(selected)
    return result, {"action_events": int(events), "unique_rows_actuated": int(changed.sum()),
                    "actions_by_edge": by_edge, "calibrated_all_edges": True,
                    "selection": "minimum_raw_space_edit_cost"}


def train(args) -> None:
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = torch.device("cuda"); seed_all(args.seed)
    diffusion, train_data, _, info, _ = load_backbone(
        args.repo, args.dataset, args.config, args.checkpoint, device
    )
    raw, edges, programs, labels, skipped, numeric_names, categorical_names = prepare_graph(
        diffusion, train_data, info, args.graph, args.calibration_target, args.min_active_rows
    )
    adapter = GraphResidualAdapter(
        programs, numeric_names, categorical_names, train_data.categories.tolist(),
        hidden=args.hidden, graph_layers=args.graph_layers, numeric_scale=args.numeric_scale,
        logit_scale=args.logit_scale, truth_temperature=args.truth_temperature,
    ).to(device)
    diffusion.eval()
    for parameter in diffusion.parameters(): parameter.requires_grad_(False)
    backbone_trainable = 0; backbone_names = []
    if args.backbone_learning_rate > 0:
        if args.rollout_steps < 2:
            raise ValueError("joint backbone training requires --rollout-steps >= 2")
        backbone_trainable, _, backbone_names = select_trainable(
            diffusion._denoise_fn, args.backbone_scope
        )
    loader = DataLoader(TensorDataset(train_data.X, labels), batch_size=args.batch_size,
                        shuffle=True, num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed))
    parameter_groups = [{"params": list(adapter.parameters()), "lr": args.learning_rate}]
    if backbone_trainable:
        parameter_groups.append({"params": [parameter for parameter in diffusion._denoise_fn.parameters()
                                              if parameter.requires_grad],
                                 "lr": args.backbone_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    history = []; started = time.perf_counter(); updates = 0
    for epoch in range(1, args.epochs + 1):
        adapter.train(); totals = {}; count = 0
        for batch, batch_labels in loader:
            batch = batch.to(device, non_blocking=True); batch_labels = batch_labels.to(device, non_blocking=True)
            prediction = unrolled_trajectory_prediction(
                diffusion, adapter, batch, args.t_min, args.t_max, args.rollout_steps
            )
            predicted_num, raw_logits, t, clean_num, clean_cat = prediction
            loss, diagnostics = adapter_loss(
                adapter, diffusion, predicted_num, raw_logits, t, clean_num, clean_cat, batch_labels,
                args.reconstruction_weight, args.applicability_weight, args.relation_weight,
                args.rate_weight, args.proximity_weight,
            )
            diffusion_anchor = loss * 0.0
            if backbone_trainable and args.diffusion_weight > 0:
                discrete, continuous = diffusion.mixed_loss(batch)
                diffusion_anchor = discrete + continuous
                loss = loss + args.diffusion_weight * diffusion_anchor
            diagnostics["diffusion_anchor"] = diffusion_anchor
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                args.grad_clip,
            )
            optimizer.step(); updates += 1
            values = {"total": loss, **diagnostics}
            for key, value in values.items(): totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(batch)
            count += len(batch)
        row = {"epoch": epoch, "updates": updates, "elapsed_seconds": time.perf_counter() - started,
               **{key: value / count for key, value in totals.items()}}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0: print(json.dumps(row), flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"adapter": adapter.state_dict(), "programs": programs,
                        "backbone": (checkpoint_state(diffusion, diffusion._denoise_fn)
                                     if backbone_trainable else None),
                        "adapter_config": {"hidden": args.hidden, "graph_layers": args.graph_layers,
                                           "numeric_scale": args.numeric_scale, "logit_scale": args.logit_scale,
                                           "truth_temperature": args.truth_temperature}},
                       args.output / f"adapter_epoch_{epoch}.pt")
            pd.DataFrame(history).to_csv(args.output / "training_curve.csv", index=False)
    (args.output / "compiled_graph.json").write_text(json.dumps(programs, indent=2))
    (args.output / "skipped_edges.json").write_text(json.dumps(skipped, indent=2))
    metadata = {"method": "topology_aware_graph_conditioned_residual_denoiser",
                "dataset": args.dataset, "seed": args.seed, "edges": len(programs),
                "targets": adapter.target_specs, "parameters": sum(p.numel() for p in adapter.parameters()),
                "backbone_frozen": not bool(backbone_trainable),
                "backbone_trainable_parameters": backbone_trainable,
                "backbone_scope": args.backbone_scope if backbone_trainable else "frozen",
                "backbone_learning_rate": args.backbone_learning_rate,
                "diffusion_anchor_weight": args.diffusion_weight,
                "epochs": args.epochs, "updates": updates,
                "trajectory_unroll_steps": args.rollout_steps,
                "t_window": [args.t_min, args.t_max], "runtime_seconds": time.perf_counter() - started,
                "weights": {name: getattr(args, f"{name}_weight") for name in
                            ("reconstruction", "applicability", "relation", "rate", "proximity")},
                "args": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}}
    (args.output / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


def sample(args) -> None:
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); sample_dir = args.output / "samples"; sample_dir.mkdir()
    device = torch.device("cuda"); seed_all(args.seed)
    diffusion, train_data, test_data, info, _ = load_backbone(
        args.repo, args.dataset, args.config, args.checkpoint, device, args.steps
    )
    raw, edges, programs, _, skipped, numeric_names, categorical_names = prepare_graph(
        diffusion, train_data, info, args.graph, args.calibration_target, args.min_active_rows
    )
    payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
    if payload.get("backbone") is not None:
        diffusion._denoise_fn.load_state_dict(payload["backbone"]["denoise_fn"], strict=True)
        diffusion.num_schedule.load_state_dict(payload["backbone"]["num_schedule"])
        diffusion.cat_schedule.load_state_dict(payload["backbone"]["cat_schedule"])
    cfg = payload["adapter_config"]
    adapter = GraphResidualAdapter(
        programs, numeric_names, categorical_names, train_data.categories.tolist(), **cfg
    ).to(device)
    adapter.load_state_dict(payload["adapter"], strict=True); adapter.eval()
    heldout = decode(train_data, test_data.X, info)
    common_seed = args.seed + 100000
    seed_all(common_seed); started = time.perf_counter()
    base_encoded = diffusion.sample(args.samples); base_runtime = time.perf_counter() - started
    base_frame = decode(train_data, base_encoded, info); base_frame.to_csv(sample_dir / "tabdiff.csv", index=False)
    base_model = diffusion._denoise_fn
    diffusion._denoise_fn = GraphConditionedModel(base_model, adapter).to(device).eval()
    seed_all(common_seed); started = time.perf_counter()
    adapted_encoded = diffusion.sample(args.samples); adapted_runtime = time.perf_counter() - started
    adapted_frame = decode(train_data, adapted_encoded, info)
    adapted_frame.to_csv(sample_dir / "graph_conditioned.csv", index=False)

    actuator = TypedActuator(edges, train_data, raw, info)
    repaired, repair_diag = calibrated_minimal_final_repair(base_frame, actuator)
    repaired.to_csv(sample_dir / "calibrated_final_repair.csv", index=False)
    adapted_repaired, adapted_repair_diag = calibrated_minimal_final_repair(adapted_frame, actuator)
    adapted_repaired.to_csv(sample_dir / "graph_conditioned_plus_repair.csv", index=False)
    rows = [
        {"method": "tabdiff", **graph_metrics(base_frame, heldout, edges), "runtime_seconds": base_runtime},
        {"method": "calibrated_final_repair", **graph_metrics(repaired, heldout, edges),
         "runtime_seconds": base_runtime, **repair_diag},
        {"method": "graph_conditioned", **graph_metrics(adapted_frame, heldout, edges),
         "runtime_seconds": adapted_runtime},
        {"method": "graph_conditioned_plus_repair", **graph_metrics(adapted_repaired, heldout, edges),
         "runtime_seconds": adapted_runtime, **adapted_repair_diag},
    ]
    pd.DataFrame(rows).to_csv(args.output / "mechanism_metrics.csv", index=False)
    (args.output / "run_metadata.json").write_text(json.dumps({
        "method": "topology_aware_graph_conditioned_residual_denoiser", "dataset": args.dataset,
        "seed": args.seed, "samples": args.samples, "steps": args.steps, "edges": len(programs),
        "skipped_edges": len(skipped), "adapter": str(args.adapter), "api_calls": 0,
    }, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", type=Path, required=True); common.add_argument("--dataset", required=True)
    common.add_argument("--config", type=Path, required=True); common.add_argument("--checkpoint", type=Path, required=True)
    common.add_argument("--graph", type=Path, required=True); common.add_argument("--output", type=Path, required=True)
    common.add_argument("--calibration-target", choices=["point", "split_median"], default="split_median")
    common.add_argument("--min-active-rows", type=int, default=64); common.add_argument("--seed", type=int, default=20261420)
    train_parser = sub.add_parser("train", parents=[common])
    train_parser.add_argument("--epochs", type=int, default=100); train_parser.add_argument("--batch-size", type=int, default=2048)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4); train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--backbone-learning-rate", type=float, default=0.0)
    train_parser.add_argument("--backbone-scope", choices=["head", "full", "detokenizer"], default="head")
    train_parser.add_argument("--diffusion-weight", type=float, default=.25,
                              help="Ordinary TabDiff loss anchoring joint end-to-end updates.")
    train_parser.add_argument("--hidden", type=int, default=128); train_parser.add_argument("--graph-layers", type=int, default=2)
    train_parser.add_argument("--numeric-scale", type=float, default=.35); train_parser.add_argument("--logit-scale", type=float, default=4.0)
    train_parser.add_argument("--truth-temperature", type=float, default=.20)
    train_parser.add_argument("--t-min", type=float, default=.05); train_parser.add_argument("--t-max", type=float, default=.65)
    train_parser.add_argument("--rollout-steps", type=int, default=1,
                              help="Differentiable truncated reverse steps; one recovers the one-step ablation.")
    train_parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    train_parser.add_argument("--applicability-weight", type=float, default=.05)
    train_parser.add_argument("--relation-weight", type=float, default=.10)
    train_parser.add_argument("--rate-weight", type=float, default=.50)
    train_parser.add_argument("--proximity-weight", type=float, default=.10)
    train_parser.add_argument("--grad-clip", type=float, default=1.0); train_parser.add_argument("--save-every", type=int, default=50)
    sample_parser = sub.add_parser("sample", parents=[common])
    sample_parser.add_argument("--adapter", type=Path, required=True); sample_parser.add_argument("--samples", type=int, default=4096)
    sample_parser.add_argument("--steps", type=int, default=50)
    return result


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    train(args) if args.command == "train" else sample(args)


if __name__ == "__main__":
    main()
