#!/usr/bin/env python3
"""Fine-tune TabDiff so predicted-clean middle states expose a typed graph.

Unlike the earlier valid-row-only graph objective, this implementation supports
mixed numerical/categorical predicates and keeps observed exceptions as
supervision.  It never turns a soft empirical relation into a hard constraint.
The objective has three independently ablatable parts:

* applicability: recover whether an edge is active for each real training row;
* relation: recover whether the active row satisfies the consequent;
* calibration: match the train-calibration conditional violation rate.

An optional two-view term makes those graph states agree across two middle
noise levels.  Checkpoints retain the ordinary TabDiff state-dict format, so
the existing sampler and row-wise dynamic guidance can evaluate them unchanged.
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
from tabdiff_probe import preprocess_dataset
from typed_dynamic_graph_guidance import edge_masks


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def update_ema(target: torch.nn.Module, source: torch.nn.Module, rate: float) -> None:
    target_values = dict(target.named_parameters())
    for name, value in source.named_parameters():
        target_values[name].mul_(rate).add_(value, alpha=1.0-rate)
    target_buffers = dict(target.named_buffers())
    for name, value in source.named_buffers():
        target_buffers[name].copy_(value)


def checkpoint_state(diffusion, model: torch.nn.Module) -> dict:
    return {
        "denoise_fn": model.state_dict(),
        "num_schedule": diffusion.num_schedule.state_dict(),
        "cat_schedule": diffusion.cat_schedule.state_dict(),
    }


def transformed_value(transform, numeric_names: list[str], raw_frame: pd.DataFrame,
                      column: str, value: float) -> float:
    anchor = raw_frame[numeric_names].apply(pd.to_numeric, errors="coerce").median().to_numpy(float)
    anchor[numeric_names.index(column)] = float(value)
    transformed = transform.transform(anchor.reshape(1, -1))
    return float(transformed[0, numeric_names.index(column)])


def compile_atom(atom: dict, raw_frame: pd.DataFrame, train_data, numeric_names: list[str],
                 categorical_names: list[str]) -> dict:
    left = atom["left_column"]
    right_kind = atom.get("right_kind", "literal")
    operator = atom["operator"]
    if left in numeric_names:
        result = {"kind": "numeric", "left_index": numeric_names.index(left), "operator": operator,
                  "tolerance": float(atom.get("tolerance", 0.0)), "left_column": left}
        if right_kind == "literal":
            result["right_kind"] = "literal"
            result["right_value"] = transformed_value(
                train_data.num_transform, numeric_names, raw_frame, left, float(atom["right_value"])
            )
            return result
        right = atom.get("right_column", "")
        if right not in numeric_names:
            raise ValueError("numeric predicates only support numeric right columns")
        result["right_kind"] = "column"
        result["right_index"] = numeric_names.index(right)
        result["right_column"] = right
        return result
    if left in categorical_names and right_kind == "literal" and operator in {"==", "!="}:
        position = categorical_names.index(left)
        categories = train_data.cat_transform.named_steps["ordinal"].categories_[position]
        matches = np.flatnonzero(np.asarray(categories).astype(str) == str(atom["right_value"]))
        if len(matches) != 1:
            raise ValueError(f"category {atom['right_value']!r} not found uniquely for {left}")
        return {"kind": "categorical", "left_index": position, "operator": operator,
                "right_kind": "literal", "right_code": int(matches[0]), "left_column": left}
    raise ValueError(f"unsupported mixed-type predicate: {atom}")


def compile_graph(graph: list[dict], raw_frame: pd.DataFrame, train_data, info: dict,
                  min_active_rows: int, allowed_classifications: set[str] | None,
                  calibration_target: str,
                  ) -> tuple[list[dict], torch.Tensor, list[dict]]:
    numeric_names = [info["column_names"][index] for index in info["num_col_idx"]]
    categorical_names = [
        info["column_names"][index]
        for index in info["target_col_idx"] + info["cat_col_idx"]
    ]
    programs: list[dict] = []
    labels: list[torch.Tensor] = []
    skipped: list[dict] = []
    for edge in graph:
        edge_id = edge.get("candidate_id", edge.get("description", "unnamed"))
        if edge.get("usage_role", "actuate") != "actuate":
            skipped.append({"candidate_id": edge_id, "reason": "usage_role"})
            continue
        if allowed_classifications and edge.get("classification") not in allowed_classifications:
            skipped.append({"candidate_id": edge_id, "reason": "classification_filter"})
            continue
        try:
            applicability = [
                compile_atom(atom, raw_frame, train_data, numeric_names, categorical_names)
                for atom in edge.get("applicability", [])
            ]
            requirements = [
                compile_atom(atom, raw_frame, train_data, numeric_names, categorical_names)
                for atom in edge.get("requirements", [])
            ]
            exceptions = [
                [compile_atom(atom, raw_frame, train_data, numeric_names, categorical_names)
                 for atom in group.get("all_of", [])]
                for group in edge.get("exceptions", [])
            ]
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append({"candidate_id": edge_id, "reason": "unsupported_predicate", "detail": str(exc)})
            continue
        if not requirements:
            skipped.append({"candidate_id": edge_id, "reason": "empty_requirements"})
            continue
        active, satisfied, violated = edge_masks(raw_frame, edge)
        if int(active.sum()) < min_active_rows:
            skipped.append({"candidate_id": edge_id, "reason": "low_active_support",
                            "support": int(active.sum())})
            continue
        point_rate = float(edge.get("calibration", {}).get(
            "violation_rate", violated.sum()/max(active.sum(), 1)
        ))
        if calibration_target == "split_median":
            split_rates = [
                float(edge[partition]["violation_rate"])
                for partition in ("discovery", "counterexample", "calibration")
                if partition in edge and np.isfinite(float(edge[partition].get("violation_rate", np.nan)))
            ]
            target_rate = float(np.median(split_rates)) if split_rates else point_rate
        else:
            target_rate = point_rate
        programs.append({
            "candidate_id": edge_id,
            "description": edge.get("description", edge_id),
            "classification": edge.get("classification", "soft"),
            "applicability": applicability,
            "requirements": requirements,
            "exceptions": exceptions,
            "active_support": int(active.sum()),
            "satisfied_support": int((active & satisfied).sum()),
            "violation_support": int(violated.sum()),
            "target_violation_rate": target_rate,
        })
        labels.append(torch.from_numpy(np.stack([active, satisfied], axis=1).astype(np.float32)))
    stacked = torch.stack(labels, dim=1) if labels else torch.empty((len(raw_frame), 0, 2))
    return programs, stacked, skipped


def categorical_probabilities(raw_logits: torch.Tensor, diffusion) -> list[torch.Tensor]:
    chunks = torch.split(raw_logits, diffusion.num_classes_w_mask, dim=-1)
    return [F.softmax(chunk[:, :int(classes)], dim=-1)
            for chunk, classes in zip(chunks, diffusion.num_classes)]


def soft_atom(atom: dict, predicted_num: torch.Tensor, cat_probs: list[torch.Tensor],
              temperature: float) -> torch.Tensor:
    if atom["kind"] == "categorical":
        probability = cat_probs[atom["left_index"]][:, atom["right_code"]]
        return probability if atom["operator"] == "==" else 1.0-probability
    left = predicted_num[:, atom["left_index"]]
    right = (torch.as_tensor(atom["right_value"], device=left.device, dtype=left.dtype)
             if atom["right_kind"] == "literal" else predicted_num[:, atom["right_index"]])
    tolerance = max(float(atom.get("tolerance", 0.0)), 0.0)
    tau = max(float(temperature), 1e-4)
    operator = atom["operator"]
    if operator in {">", ">="}:
        return torch.sigmoid((left-right+tolerance)/tau)
    if operator in {"<", "<="}:
        return torch.sigmoid((right-left+tolerance)/tau)
    equal = torch.exp(-0.5*((left-right)/max(tau+tolerance, 1e-4)).square())
    if operator == "==":
        return equal
    if operator == "!=":
        return 1.0-equal
    raise ValueError(operator)


def soft_graph_state(predicted_num: torch.Tensor, raw_logits: torch.Tensor, diffusion,
                     programs: list[dict], temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    cat_probs = categorical_probabilities(raw_logits, diffusion)
    gates, satisfaction = [], []
    for program in programs:
        gate = torch.ones(len(predicted_num), device=predicted_num.device, dtype=predicted_num.dtype)
        for atom in program["applicability"]:
            gate = gate*soft_atom(atom, predicted_num, cat_probs, temperature)
        # Exception groups are ORs of conjunctions.  The product below is the
        # differentiable probability that no exception group applies.
        not_exception = torch.ones_like(gate)
        for group in program["exceptions"]:
            group_truth = torch.ones_like(gate)
            for atom in group:
                group_truth = group_truth*soft_atom(atom, predicted_num, cat_probs, temperature)
            not_exception = not_exception*(1.0-group_truth)
        gate = gate*not_exception
        consequent = torch.ones_like(gate)
        for atom in program["requirements"]:
            consequent = consequent*soft_atom(atom, predicted_num, cat_probs, temperature)
        gates.append(gate.clamp(1e-5, 1.0-1e-5))
        satisfaction.append(consequent.clamp(1e-5, 1.0-1e-5))
    return torch.stack(gates, dim=1), torch.stack(satisfaction, dim=1)


def predict_middle(diffusion, batch: torch.Tensor, t_min: float, t_max: float
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(batch)
    x_num = batch[:, :diffusion.num_numerical_features]
    x_cat = batch[:, diffusion.num_numerical_features:].long()
    t = t_min+(t_max-t_min)*torch.rand(batch_size, 1, device=batch.device, dtype=x_num.dtype)
    sigma_num = diffusion.num_schedule.total_noise(t)
    sigma_cat = diffusion.cat_schedule.total_noise(t)
    noisy_num = x_num+torch.randn_like(x_num)*sigma_num
    move_chance = -torch.expm1(-sigma_cat)
    if x_cat.shape[1] > 0:
        strategy = "soft" if diffusion.cat_scheduler == "log_linear_per_column" else "hard"
        _, noisy_cat = diffusion.q_xt(x_cat, move_chance, strategy=strategy)
    else:
        noisy_cat = x_cat
    predicted_num, raw_logits = diffusion._denoise_fn(
        noisy_num, noisy_cat, t.squeeze(-1), sigma=sigma_num
    )
    return predicted_num, raw_logits, t


def calibrated_graph_loss(diffusion, batch: torch.Tensor, labels: torch.Tensor,
                          programs: list[dict], t_min: float, t_max: float,
                          temperature: float, applicability_weight: float,
                          relation_weight: float, rate_weight: float,
                          consistency_weight: float) -> tuple[torch.Tensor, dict]:
    predicted_num, raw_logits, _ = predict_middle(diffusion, batch, t_min, t_max)
    gates, satisfaction = soft_graph_state(
        predicted_num, raw_logits, diffusion, programs, temperature
    )
    clean_active = labels[:, :, 0]
    clean_satisfied = labels[:, :, 1]
    applicability_loss = F.binary_cross_entropy(gates, clean_active)
    active_support = clean_active.sum().clamp_min(1.0)
    relation_terms = F.binary_cross_entropy(satisfaction, clean_satisfied, reduction="none")
    relation_loss = (relation_terms*clean_active).sum()/active_support
    target_rates = torch.as_tensor(
        [program["target_violation_rate"] for program in programs],
        device=batch.device, dtype=predicted_num.dtype,
    )
    predicted_rates = (gates*(1.0-satisfaction)).sum(dim=0)/gates.sum(dim=0).clamp_min(1e-5)
    rate_loss = F.smooth_l1_loss(predicted_rates, target_rates, beta=.02)
    consistency_loss = predicted_num.sum()*0.0
    if consistency_weight > 0:
        second_num, second_logits, _ = predict_middle(diffusion, batch, t_min, t_max)
        second_gates, second_satisfaction = soft_graph_state(
            second_num, second_logits, diffusion, programs, temperature
        )
        consistency_loss = F.mse_loss(gates, second_gates)+F.mse_loss(satisfaction, second_satisfaction)
    total = (applicability_weight*applicability_loss+relation_weight*relation_loss+
             rate_weight*rate_loss+consistency_weight*consistency_loss)
    diagnostics = {
        "applicability": applicability_loss,
        "relation": relation_loss,
        "rate": rate_loss,
        "consistency": consistency_loss,
        "predicted_rate_mean": predicted_rates.mean(),
        "target_rate_mean": target_rates.mean(),
    }
    return total, diagnostics


def select_trainable(model: torch.nn.Module, scope: str) -> tuple[int, int, list[str]]:
    selected = []
    for name, parameter in model.named_parameters():
        if scope == "full":
            enabled = True
        elif scope == "head":
            enabled = ".decoder." in name or ".detokenizer." in name
        elif scope == "detokenizer":
            enabled = ".detokenizer." in name
        else:
            raise ValueError(scope)
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(name)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise RuntimeError(f"scope {scope!r} selected no parameters")
    return trainable, total, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--trainable-scope", choices=["full", "head", "detokenizer"], default="head")
    parser.add_argument("--applicability-weight", type=float, default=.10)
    parser.add_argument("--relation-weight", type=float, default=.10)
    parser.add_argument("--rate-weight", type=float, default=2.0)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--t-min", type=float, default=.10)
    parser.add_argument("--t-max", type=float, default=.50)
    parser.add_argument("--truth-temperature", type=float, default=.20)
    parser.add_argument("--min-active-rows", type=int, default=64)
    parser.add_argument("--calibration-target", choices=["point", "split_median"], default="point",
                        help="Train-only target for each soft edge's conditional violation rate.")
    parser.add_argument("--allowed-classifications", nargs="+", choices=["hard", "near_hard", "soft"])
    parser.add_argument("--ema-decay", type=float, default=.997)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261390)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    if not 0 <= args.t_min < args.t_max <= 1:
        raise ValueError("t window must lie in [0,1]")
    seed_all(args.seed)
    device = torch.device("cuda")
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    train_data, _ = preprocess_dataset(args.repo, args.dataset)
    info = json.loads((args.repo/"data"/args.dataset/"info.json").read_text())
    raw_frame = decode(train_data, train_data.X, info)
    graph = json.loads(args.graph.read_text())
    allowed = set(args.allowed_classifications) if args.allowed_classifications else None
    programs, labels, skipped = compile_graph(
        graph, raw_frame, train_data, info, args.min_active_rows, allowed, args.calibration_target
    )
    graph_weight_sum = (args.applicability_weight+args.relation_weight+
                        args.rate_weight+args.consistency_weight)
    if graph_weight_sum > 0 and not programs:
        raise RuntimeError("no differentiable mixed-type graph programs")
    loader = DataLoader(
        TensorDataset(train_data.X, labels), batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(args.seed),
    )
    with args.config.open("rb") as handle:
        config = copy.deepcopy(pickle.load(handle))
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories+1).tolist()
    model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["denoise_fn"], strict=True)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params["scheduler"] = "power_mean_per_column"
    diffusion_params["cat_scheduler"] = "log_linear_per_column"
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, model, None,
        **diffusion_params, device=device,
    ).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    for schedule in (diffusion.num_schedule, diffusion.cat_schedule):
        for parameter in schedule.parameters():
            parameter.requires_grad_(False)
    trainable, total_parameters, selected_names = select_trainable(
        diffusion._denoise_fn, args.trainable_scope
    )
    ema_model = copy.deepcopy(diffusion._denoise_fn).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in diffusion._denoise_fn.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in diffusion._denoise_fn.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=0,
    )

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config["mixed_graph_finetune"] = serializable_args | {"compiled_edges": len(programs)}
    with (args.output/"config.pkl").open("wb") as handle:
        pickle.dump(config, handle)
    (args.output/"compiled_graph.json").write_text(json.dumps(programs, indent=2))
    (args.output/"skipped_edges.json").write_text(json.dumps(skipped, indent=2))
    (args.output/"trainable_parameters.json").write_text(json.dumps(selected_names, indent=2))

    rows = []
    started = time.perf_counter()
    updates = 0
    keys = ["discrete", "continuous", "graph", "applicability", "relation", "rate",
            "consistency", "predicted_rate_mean", "target_rate_mean", "total"]
    for epoch in range(1, args.epochs+1):
        diffusion.train()
        sums = {key: 0.0 for key in keys}
        count = 0
        for batch, graph_labels in loader:
            batch = batch.to(device, non_blocking=True)
            graph_labels = graph_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            discrete, continuous = diffusion.mixed_loss(batch)
            auxiliary = torch.zeros((), device=device)
            diagnostics = {key: auxiliary for key in keys[3:-1]}
            if graph_weight_sum > 0:
                auxiliary, diagnostics = calibrated_graph_loss(
                    diffusion, batch, graph_labels, programs, args.t_min, args.t_max,
                    args.truth_temperature, args.applicability_weight, args.relation_weight,
                    args.rate_weight, args.consistency_weight,
                )
            total = discrete+continuous+auxiliary
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in diffusion._denoise_fn.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            update_ema(ema_model, diffusion._denoise_fn, args.ema_decay)
            updates += 1
            values = {"discrete": discrete, "continuous": continuous, "graph": auxiliary,
                      "total": total, **diagnostics}
            for key in keys:
                sums[key] += float(values[key].detach())*len(batch)
            count += len(batch)
        row = {"epoch": epoch, **{f"{key}_loss" if key not in {"predicted_rate_mean", "target_rate_mean"}
                                         else key: value/count for key, value in sums.items()},
               "learning_rate": optimizer.param_groups[0]["lr"], "updates": updates,
               "elapsed_seconds": time.perf_counter()-started}
        rows.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(checkpoint_state(diffusion, diffusion._denoise_fn),
                       args.output/f"model_epoch_{epoch}.pt")
            torch.save(checkpoint_state(diffusion, ema_model),
                       args.output/f"ema_model_epoch_{epoch}.pt")
            pd.DataFrame(rows).to_csv(args.output/"training_curve.csv", index=False)

    squared = 0.0
    initial_squared = 0.0
    for name, value in diffusion._denoise_fn.state_dict().items():
        if value.is_floating_point():
            squared += float((value.detach().cpu()-initial[name]).square().sum())
            initial_squared += float(initial[name].square().sum())
    metadata = {
        "status": "executed", "dataset": args.dataset, "epochs": args.epochs, "updates": updates,
        "objective": "mixed_type_applicability_relation_rate_calibration",
        "weights": {"applicability": args.applicability_weight, "relation": args.relation_weight,
                    "rate": args.rate_weight, "consistency": args.consistency_weight},
        "t_window": [args.t_min, args.t_max], "trainable_scope": args.trainable_scope,
        "calibration_target": args.calibration_target,
        "trainable_parameters": trainable, "total_parameters": total_parameters,
        "compiled_edges": len(programs), "skipped_edges": len(skipped),
        "noise_schedules_frozen": True, "observed_exceptions_preserved": True,
        "relative_parameter_displacement": math.sqrt(squared/max(initial_squared, 1e-30)),
        "runtime_seconds": time.perf_counter()-started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device), "seed": args.seed,
    }
    (args.output/"training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
