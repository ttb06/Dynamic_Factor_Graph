#!/usr/bin/env python3
"""End-to-end trajectory-teacher fine-tuning for typed-graph TabDiff.

The frozen backbone supplies an x0 preview at a sampled middle noise level.
An executable, train-verified graph produces a detached corrected target for
that preview.  The student is then optimized with the ordinary TabDiff loss
plus one of several conservative distillation objectives.  This differs from
the older fuzzy satisfaction loss: it teaches a concrete action while an
unchanged-coordinate anchor explicitly protects the pretrained distribution.

No held-out row or evaluation metric is used to construct a target.
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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from conditioned_tabdiff_adapter import decode
from dense_numeric_factor_messages import DenseNumericFactors
from finetune_mixed_graph_calibrated_tabdiff import checkpoint_state, select_trainable
from tabdiff_probe import preprocess_dataset


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def update_ema(target: torch.nn.Module, source: torch.nn.Module, rate: float) -> None:
    target_parameters = dict(target.named_parameters())
    for name, value in source.named_parameters():
        target_parameters[name].mul_(rate).add_(value, alpha=1.0-rate)
    target_buffers = dict(target.named_buffers())
    for name, value in source.named_buffers():
        target_buffers[name].copy_(value)


def categorical_codes(x_cat: torch.Tensor, logits: torch.Tensor,
                      category_sizes: list[int]) -> torch.Tensor:
    if not category_sizes:
        return torch.empty((len(x_cat), 0), dtype=torch.long, device=x_cat.device)
    widths = [size+1 for size in category_sizes]
    current_chunks = torch.split(x_cat, widths, dim=-1)
    logit_chunks = torch.split(logits, widths, dim=-1)
    codes = []
    for size, current, prediction in zip(category_sizes, current_chunks, logit_chunks):
        current_code = current.argmax(dim=-1)
        predicted_code = prediction[:, :size].argmax(dim=-1)
        codes.append(torch.where(current_code.eq(size), predicted_code, current_code))
    return torch.stack(codes, dim=1)


def noisy_view(diffusion, batch: torch.Tensor, t_min: float, t_max: float,
               teacher: torch.nn.Module) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    clean_num = batch[:, :diffusion.num_numerical_features]
    clean_cat = batch[:, diffusion.num_numerical_features:].long()
    t = t_min+(t_max-t_min)*torch.rand(
        batch_size, 1, device=batch.device, dtype=clean_num.dtype)
    sigma_num = diffusion.num_schedule.total_noise(t)
    sigma_cat = diffusion.cat_schedule.total_noise(t)
    noisy_num = clean_num+torch.randn_like(clean_num)*sigma_num
    move_chance = -torch.expm1(-sigma_cat)
    if clean_cat.shape[1]:
        strategy = "soft" if diffusion.cat_scheduler == "log_linear_per_column" else "hard"
        _, noisy_cat = diffusion.q_xt(clean_cat, move_chance, strategy=strategy)
    else:
        noisy_cat = clean_cat
    with torch.no_grad():
        teacher_num, teacher_logits = teacher(
            noisy_num, noisy_cat, t.squeeze(-1), sigma=sigma_num)
    student_num, student_logits = diffusion._denoise_fn(
        noisy_num, noisy_cat, t.squeeze(-1), sigma=sigma_num)
    return {
        "student_num": student_num, "student_logits": student_logits,
        "teacher_num": teacher_num.detach(), "teacher_logits": teacher_logits.detach(),
        "clean_num": clean_num, "noisy_cat": noisy_cat, "t": t.squeeze(-1),
    }


@torch.no_grad()
def graph_teacher_target(view: dict[str, torch.Tensor], train_data, info: dict,
                         factors: DenseNumericFactors, strength: float,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    sizes = [int(value) for value in train_data.categories]
    codes = categorical_codes(view["noisy_cat"], view["teacher_logits"], sizes)
    encoded = torch.cat([view["teacher_num"], codes.to(view["teacher_num"].dtype)], dim=1)
    frame = decode(train_data, encoded.cpu(), info)
    corrected, changed, events = factors.project_violations(frame, strength)
    target = torch.as_tensor(
        factors.encode_numeric(corrected), dtype=view["teacher_num"].dtype, device=device)
    changed_tensor = torch.as_tensor(changed, dtype=torch.bool, device=device)
    return target, changed_tensor, int(events)


def masked_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values*weights).sum()/weights.sum().clamp_min(1.0)


def distillation_terms(view: dict[str, torch.Tensor], target: torch.Tensor,
                       changed: torch.Tensor, target_mode: str,
                       t_min: float, t_max: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student = view["student_num"]
    teacher = view["teacher_num"]
    # Mid-peak weighting is per row and is normalized to preserve the nominal
    # auxiliary coefficient across different time windows.
    progress = (view["t"]-t_min)/max(t_max-t_min, 1e-8)
    row_weight = (.25+.75*(4.0*progress*(1.0-progress)).clamp(0.0, 1.0)).unsqueeze(1)
    element = F.smooth_l1_loss(student, target, beta=.05, reduction="none")*row_weight
    if target_mode == "all":
        graph = element.mean()
    elif target_mode == "sparse":
        graph = masked_average(element, changed)
    elif target_mode == "delta_normalized":
        target_delta = target-teacher
        scale = target_delta.abs().mean(dim=0, keepdim=True).clamp_min(.05)
        delta_error = F.smooth_l1_loss(
            (student-teacher)/scale, target_delta/scale, beta=.10, reduction="none")*row_weight
        graph = masked_average(delta_error, changed)
    else:
        raise ValueError(target_mode)
    unchanged = ~changed
    anchor = masked_average((student-teacher).square()*row_weight, unchanged)
    correction = student-teacher
    return graph, anchor, correction


def pcgrad_step(base_loss: torch.Tensor, auxiliary: torch.Tensor, weight: float,
                parameters: list[torch.nn.Parameter]) -> tuple[float, bool]:
    base_gradients = torch.autograd.grad(
        base_loss, parameters, retain_graph=True, allow_unused=True)
    auxiliary_gradients = torch.autograd.grad(
        auxiliary, parameters, allow_unused=True)
    dot = torch.zeros((), device=base_loss.device)
    norm = torch.zeros((), device=base_loss.device)
    for base, extra in zip(base_gradients, auxiliary_gradients):
        if base is not None and extra is not None:
            dot = dot+(base*extra).sum()
            norm = norm+(base*base).sum()
    conflict = bool(dot.detach() < 0)
    coefficient = (dot/norm.clamp_min(1e-20)).detach() if conflict else dot.detach()*0.0
    for parameter, base, extra in zip(parameters, base_gradients, auxiliary_gradients):
        if base is None and extra is None:
            parameter.grad = None
            continue
        if base is None:
            combined = weight*extra
        elif extra is None:
            combined = base
        else:
            projected = extra-coefficient*base if conflict else extra
            combined = base+weight*projected
        parameter.grad = combined.detach()
    return float(dot.detach()), conflict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="news")
    parser.add_argument("--regression-target-mode", choices=["categorical", "numerical"],
                        default="numerical")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--trainable-scope", choices=["full", "head", "detokenizer"], default="head")
    parser.add_argument("--target-mode", choices=["sparse", "all", "delta_normalized"],
                        default="sparse")
    parser.add_argument("--teacher-transport", choices=["direct", "rank_order"], default="direct")
    parser.add_argument("--include-relation-families", nargs="+",
                        help="Optional allow-list for graph edge families.")
    parser.add_argument("--exclude-relation-families", nargs="+",
                        help="Optional deny-list for graph edge families.")
    parser.add_argument("--target-control", choices=["none", "row_roll", "sign_flip"],
                        default="none",
                        help="Matched-action controls used to test whether graph structure matters.")
    parser.add_argument("--benefit-filter", choices=["none", "cell", "row"], default="none",
                        help="Train-only filter retaining graph actions that reduce x0 recovery error.")
    parser.add_argument("--teacher-strength", type=float, default=.50)
    parser.add_argument("--auxiliary-weight", type=float, default=.10)
    parser.add_argument("--anchor-weight", type=float, default=.50)
    parser.add_argument("--trajectory-weight", type=float, default=0.0)
    parser.add_argument("--views", type=int, choices=[1, 2], default=1)
    parser.add_argument("--stable-only", action="store_true")
    parser.add_argument("--gradient-mode", choices=["sum", "pcgrad"], default="sum")
    parser.add_argument("--t-min", type=float, default=.10)
    parser.add_argument("--t-max", type=float, default=.40)
    parser.add_argument("--ema-decay", type=float, default=.997)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    if args.stable_only and args.views != 2:
        raise ValueError("stable-only requires two views")
    seed_all(args.seed)
    device = torch.device("cuda")
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    train_data, _ = preprocess_dataset(
        args.repo, args.dataset, regression_target_mode=args.regression_target_mode)
    info = json.loads((args.repo/"data"/args.dataset/"info.json").read_text())
    raw_train = decode(train_data, train_data.X, info)
    edges = copy.deepcopy(json.loads(args.graph.read_text()))
    if args.include_relation_families:
        allowed = set(args.include_relation_families)
        edges = [edge for edge in edges if edge.get("relation_family") in allowed]
    if args.exclude_relation_families:
        denied = set(args.exclude_relation_families)
        edges = [edge for edge in edges if edge.get("relation_family") not in denied]
    if not edges:
        raise RuntimeError("edge-family filters removed the entire graph")
    if args.teacher_transport == "rank_order":
        for edge in edges:
            if edge.get("relation_family") == "statistical_order_program" and edge.get("program"):
                edge["program"]["transport"] = "rank"
    factors = DenseNumericFactors(
        train_data, raw_train, info, edges, args.seed, trees=0, min_r2=0.0)

    with args.config.open("rb") as handle:
        config = copy.deepcopy(pickle.load(handle))
    parameters = copy.deepcopy(config["unimodmlp_params"])
    parameters["d_numerical"] = train_data.d_numerical
    parameters["categories"] = (train_data.categories+1).tolist()
    model = Model(UniModMLP(**parameters), **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["denoise_fn"], strict=True)
    diffusion_parameters = copy.deepcopy(config["diffusion_params"])
    diffusion_parameters.update(
        scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, model, None,
        **diffusion_parameters, device=device).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    for schedule in (diffusion.num_schedule, diffusion.cat_schedule):
        for parameter in schedule.parameters():
            parameter.requires_grad_(False)
    teacher = copy.deepcopy(diffusion._denoise_fn).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainable_count, total_count, selected_names = select_trainable(
        diffusion._denoise_fn, args.trainable_scope)
    trainable_parameters = [
        parameter for parameter in diffusion._denoise_fn.parameters() if parameter.requires_grad]
    ema_model = copy.deepcopy(diffusion._denoise_fn).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.0)
    loader = DataLoader(
        TensorDataset(train_data.X), batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed))

    serializable = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output/"experiment_config.json").write_text(json.dumps(serializable, indent=2))
    (args.output/"trainable_parameters.json").write_text(json.dumps(selected_names, indent=2))
    rows: list[dict] = []
    started = time.perf_counter()
    updates = 0
    total_events = 0
    total_changed = 0
    total_proposed = 0
    conflicts = 0
    for epoch in range(1, args.epochs+1):
        diffusion.train()
        sums = {key: 0.0 for key in (
            "discrete", "continuous", "graph", "anchor", "trajectory", "auxiliary", "total")}
        seen = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            discrete, continuous = diffusion.mixed_loss(batch)
            base_loss = discrete+continuous
            views = [noisy_view(diffusion, batch, args.t_min, args.t_max, teacher)
                     for _ in range(args.views)]
            targets, masks = [], []
            for view in views:
                target, mask, events = graph_teacher_target(
                    view, train_data, info, factors, args.teacher_strength, device)
                if args.target_control == "row_roll":
                    # Keep the exact correction-magnitude distribution but
                    # assign each action to the next row.  torch.roll is
                    # deterministic and consumes no training RNG, so the
                    # ordinary diffusion-loss stream remains paired.
                    delta = target-view["teacher_num"]
                    target = view["teacher_num"]+torch.roll(delta, shifts=1, dims=0)
                    mask = torch.roll(mask, shifts=1, dims=0)
                elif args.target_control == "sign_flip":
                    target = view["teacher_num"]-(target-view["teacher_num"])
                total_proposed += int(mask.sum())
                if args.benefit_filter != "none":
                    clean = view["clean_num"]
                    benefit = (view["teacher_num"]-clean).square()-(target-clean).square()
                    if args.benefit_filter == "cell":
                        mask = mask & (benefit > 0)
                    else:
                        row_benefit = (benefit*mask.to(benefit.dtype)).sum(dim=1) > 0
                        mask = mask & row_benefit.unsqueeze(1)
                targets.append(target); masks.append(mask)
                total_events += events
            if args.stable_only:
                first_delta = targets[0]-views[0]["teacher_num"]
                second_delta = targets[1]-views[1]["teacher_num"]
                stable = masks[0] & masks[1] & (first_delta*second_delta >= 0)
                masks = [stable, stable]
            total_changed += sum(int(mask.sum()) for mask in masks)
            graph_terms, anchor_terms, corrections = [], [], []
            for view, target, mask in zip(views, targets, masks):
                graph, anchor, correction = distillation_terms(
                    view, target, mask, args.target_mode, args.t_min, args.t_max)
                graph_terms.append(graph); anchor_terms.append(anchor); corrections.append(correction)
            graph_loss = torch.stack(graph_terms).mean()
            anchor_loss = torch.stack(anchor_terms).mean()
            trajectory_loss = graph_loss*0.0
            if len(corrections) == 2 and args.trajectory_weight > 0:
                shared = masks[0] & masks[1]
                # Matching the action across two independently noised previews
                # encourages a persistent edge state without forcing x0 itself
                # to be identical at different noise levels.
                trajectory_loss = masked_average(
                    F.smooth_l1_loss(corrections[0], corrections[1].detach(),
                                     beta=.05, reduction="none"), shared)
            auxiliary = (graph_loss+args.anchor_weight*anchor_loss
                         +args.trajectory_weight*trajectory_loss)
            if args.auxiliary_weight <= 0:
                base_loss.backward()
                dot, conflict = math.nan, False
            elif args.gradient_mode == "pcgrad":
                dot, conflict = pcgrad_step(
                    base_loss, auxiliary, args.auxiliary_weight, trainable_parameters)
                conflicts += int(conflict)
            else:
                total = base_loss+args.auxiliary_weight*auxiliary
                total.backward(); dot, conflict = math.nan, False
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step(); update_ema(ema_model, diffusion._denoise_fn, args.ema_decay)
            updates += 1
            total = base_loss+args.auxiliary_weight*auxiliary
            values = {"discrete": discrete, "continuous": continuous, "graph": graph_loss,
                      "anchor": anchor_loss, "trajectory": trajectory_loss,
                      "auxiliary": auxiliary, "total": total}
            for key, value in values.items():
                sums[key] += float(value.detach())*len(batch)
            seen += len(batch)
        row = {"epoch": epoch, **{f"{key}_loss": value/seen for key, value in sums.items()},
               "updates": updates, "gradient_conflicts": conflicts,
               "elapsed_seconds": time.perf_counter()-started}
        rows.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(args.output/"training_curve.csv", index=False)

    torch.save(checkpoint_state(diffusion, diffusion._denoise_fn), args.output/"model_final.pt")
    torch.save(checkpoint_state(diffusion, ema_model), args.output/"ema_model_final.pt")
    metadata = {
        "status": "executed", "dataset": args.dataset, "objective": "trajectory_teacher_distillation",
        "epochs": args.epochs, "updates": updates, "trainable_scope": args.trainable_scope,
        "trainable_parameters": trainable_count, "total_parameters": total_count,
        "target_mode": args.target_mode, "teacher_transport": args.teacher_transport,
        "target_control": args.target_control,
        "benefit_filter": args.benefit_filter,
        "relation_family_filter": {
            "include": args.include_relation_families, "exclude": args.exclude_relation_families},
        "compiled_edges": len(edges),
        "views": args.views, "stable_only": args.stable_only,
        "gradient_mode": args.gradient_mode,
        "gradient_conflict_fraction": conflicts/max(updates, 1),
        "teacher_projection_events": total_events,
        "teacher_proposed_cells": total_proposed,
        "teacher_changed_cells": total_changed,
        "beneficial_cell_retention": total_changed/max(total_proposed, 1),
        "runtime_seconds": time.perf_counter()-started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device), "seed": args.seed,
    }
    (args.output/"training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
