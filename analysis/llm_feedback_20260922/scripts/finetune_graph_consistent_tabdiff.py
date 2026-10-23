#!/usr/bin/env python3
"""End-to-end TabDiff fine-tuning with a trajectory-level typed-graph loss.

The auxiliary objective is deliberately conservative: it is evaluated only on
training rows whose clean values already satisfy a semantically authorized
edge.  Observed exceptions are not relabeled or forced to obey the rule.  The
entire denoiser is trainable; learned noise schedules remain frozen so they
cannot game the graph objective.
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
from torch.utils.data import DataLoader, TensorDataset

from conditioned_tabdiff_adapter import decode
from tabdiff_probe import preprocess_dataset
from typed_dynamic_graph_guidance import edge_masks


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_ema(target: torch.nn.Module, source: torch.nn.Module, rate: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.detach().mul_(rate).add_(source_param.detach(), alpha=1-rate)


def checkpoint_state(diffusion, model: torch.nn.Module) -> dict:
    return {
        "denoise_fn": model.state_dict(),
        "num_schedule": diffusion.num_schedule.state_dict(),
        "cat_schedule": diffusion.cat_schedule.state_dict(),
    }


def transformed_threshold(transform, numeric_names: list[str], raw_frame: pd.DataFrame,
                          column: str, value: float) -> float:
    anchor = raw_frame[numeric_names].apply(pd.to_numeric, errors="coerce").median().to_numpy(float)
    anchor[numeric_names.index(column)] = float(value)
    return float(transform.transform(anchor.reshape(1, -1))[0, numeric_names.index(column)])


def compile_graph(graph: list[dict], raw_frame: pd.DataFrame, train_data, info: dict,
                  min_valid_rows: int, allowed_classifications: set[str] | None = None
                  ) -> tuple[list[dict], torch.Tensor, list[dict]]:
    numeric_names = [info["column_names"][index] for index in info["num_col_idx"]]
    numeric_set = set(numeric_names)
    programs, clean_valid, skipped = [], [], []
    for edge in graph:
        if edge.get("usage_role", "actuate") != "actuate":
            continue
        if allowed_classifications and edge.get("classification") not in allowed_classifications:
            skipped.append({"candidate_id": edge["candidate_id"], "reason": "classification_filter",
                            "classification": edge.get("classification")})
            continue
        atoms = edge.get("applicability", []) + edge.get("requirements", [])
        unsupported = [atom for atom in atoms if atom.get("right_kind") != "literal"
                       or atom.get("left_column") not in numeric_set]
        if unsupported:
            skipped.append({"candidate_id": edge["candidate_id"], "reason": "nonliteral_or_nonnumeric_atom"})
            continue
        active, satisfied, _ = edge_masks(raw_frame, edge)
        valid = active & satisfied
        if int(valid.sum()) < min_valid_rows:
            skipped.append({"candidate_id": edge["candidate_id"], "reason": "low_clean_valid_support",
                            "support": int(valid.sum())})
            continue
        requirements = []
        for atom in edge.get("requirements", []):
            requirements.append({
                "column": atom["left_column"],
                "index": numeric_names.index(atom["left_column"]),
                "operator": atom["operator"],
                "threshold": transformed_threshold(
                    train_data.num_transform, numeric_names, raw_frame,
                    atom["left_column"], float(atom["right_value"]),
                ),
            })
        programs.append({
            "candidate_id": edge["candidate_id"], "description": edge["description"],
            "classification": edge["classification"], "requirements": requirements,
            "clean_valid_support": int(valid.sum()),
        })
        clean_valid.append(torch.from_numpy(valid.astype(np.float32)))
    labels = torch.stack(clean_valid, dim=1) if clean_valid else torch.empty((len(raw_frame), 0))
    return programs, labels, skipped


def fuzzy_truth(prediction: torch.Tensor, operator: str, threshold: float, tau: float) -> torch.Tensor:
    value = torch.as_tensor(threshold, device=prediction.device, dtype=prediction.dtype)
    if operator in {">", ">="}:
        return torch.sigmoid((prediction-value)/tau)
    if operator in {"<", "<="}:
        return torch.sigmoid((value-prediction)/tau)
    equal = torch.exp(-.5*((prediction-value)/tau).square())
    if operator == "==":
        return equal
    if operator == "!=":
        return 1-equal
    raise ValueError(operator)


def graph_loss(diffusion, batch: torch.Tensor, labels: torch.Tensor, programs: list[dict],
               t_min: float, t_max: float, tau: float) -> tuple[torch.Tensor, dict]:
    batch_size = len(batch); device = batch.device
    x_num = batch[:, :diffusion.num_numerical_features]
    x_cat = batch[:, diffusion.num_numerical_features:].long()
    t = t_min + (t_max-t_min)*torch.rand(batch_size, 1, device=device, dtype=x_num.dtype)
    sigma_num = diffusion.num_schedule.total_noise(t)
    sigma_cat = diffusion.cat_schedule.total_noise(t)
    noisy_num = x_num + torch.randn_like(x_num)*sigma_num
    move_chance = -torch.expm1(-sigma_cat)
    if x_cat.shape[1] > 0:
        strategy = "soft" if diffusion.cat_scheduler == "log_linear_per_column" else "hard"
        _, noisy_cat = diffusion.q_xt(x_cat, move_chance, strategy=strategy)
    else:
        noisy_cat = x_cat
    predicted_num, _ = diffusion._denoise_fn(noisy_num, noisy_cat, t.squeeze(-1), sigma=sigma_num)
    losses, supports = [], {}
    for edge_index, program in enumerate(programs):
        eligible = labels[:, edge_index]
        support = eligible.sum().clamp_min(1.0)
        truth = torch.ones(batch_size, device=device, dtype=predicted_num.dtype)
        for requirement in program["requirements"]:
            truth = truth*fuzzy_truth(predicted_num[:, requirement["index"]], requirement["operator"],
                                      requirement["threshold"], tau)
        edge_loss = (eligible*(1-truth)).sum()/support
        weight = 1.0 if program["classification"] == "hard" else .5
        losses.append(weight*edge_loss)
        supports[program["candidate_id"]] = int(eligible.detach().sum())
    if not losses:
        return predicted_num.sum()*0, supports
    return torch.stack(losses).mean(), supports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="shoppers")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--graph-weight", type=float, default=.05)
    parser.add_argument("--t-min", type=float, default=.15)
    parser.add_argument("--t-max", type=float, default=.60)
    parser.add_argument("--truth-temperature", type=float, default=.20)
    parser.add_argument("--min-valid-rows", type=int, default=64)
    parser.add_argument("--allowed-classifications", nargs="+", choices=["hard", "near_hard", "soft"],
                        help="Optional semantic hardness filter; default uses all actuate edges.")
    parser.add_argument("--ema-decay", type=float, default=.997)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261230)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    if not 0 <= args.t_min < args.t_max <= 1:
        raise ValueError("t window must lie in [0,1]")
    seed_all(args.seed); device = torch.device("cuda"); sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    train_data, _ = preprocess_dataset(args.repo, args.dataset)
    info = json.loads((args.repo/"data"/args.dataset/"info.json").read_text())
    raw_frame = decode(train_data, train_data.X, info)
    graph = json.loads(args.graph.read_text())
    allowed = set(args.allowed_classifications) if args.allowed_classifications else None
    programs, labels, skipped = compile_graph(
        graph, raw_frame, train_data, info, args.min_valid_rows, allowed
    )
    if args.graph_weight > 0 and not programs:
        raise RuntimeError("no differentiable graph programs")
    loader = DataLoader(TensorDataset(train_data.X, labels), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed))
    with args.config.open("rb") as handle:
        config = copy.deepcopy(pickle.load(handle))
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories+1).tolist()
    model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["denoise_fn"], strict=True)
    dp = copy.deepcopy(config["diffusion_params"])
    dp["scheduler"] = "power_mean_per_column"; dp["cat_scheduler"] = "log_linear_per_column"
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, model, None,
                                      **dp, device=device).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    for schedule in (diffusion.num_schedule, diffusion.cat_schedule):
        for parameter in schedule.parameters():
            parameter.requires_grad_(False)
    ema_model = copy.deepcopy(diffusion._denoise_fn).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    initial = {name: value.detach().cpu().clone() for name, value in diffusion._denoise_fn.state_dict().items()}
    optimizer = torch.optim.AdamW(diffusion._denoise_fn.parameters(), lr=args.learning_rate, weight_decay=0)

    config["graph_finetune"] = vars(args) | {
        "config": str(args.config), "checkpoint": str(args.checkpoint), "graph": str(args.graph),
        "output": str(args.output), "compiled_edges": len(programs),
    }
    with (args.output/"config.pkl").open("wb") as handle:
        pickle.dump(config, handle)
    (args.output/"compiled_graph.json").write_text(json.dumps(programs, indent=2))
    (args.output/"skipped_edges.json").write_text(json.dumps(skipped, indent=2))

    rows=[]; started=time.perf_counter(); updates=0
    for epoch in range(1, args.epochs+1):
        diffusion.train(); sums={"discrete":0.0,"continuous":0.0,"graph":0.0,"total":0.0}; count=0
        for batch, valid_labels in loader:
            batch=batch.to(device, non_blocking=True); valid_labels=valid_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            discrete, continuous = diffusion.mixed_loss(batch)
            auxiliary = torch.zeros((), device=device)
            if args.graph_weight > 0:
                auxiliary, _ = graph_loss(diffusion, batch, valid_labels, programs,
                                          args.t_min, args.t_max, args.truth_temperature)
            total = discrete+continuous+args.graph_weight*auxiliary
            total.backward(); torch.nn.utils.clip_grad_norm_(diffusion._denoise_fn.parameters(), 1.0)
            optimizer.step(); update_ema(ema_model, diffusion._denoise_fn, args.ema_decay); updates+=1
            for key,value in (("discrete",discrete),("continuous",continuous),("graph",auxiliary),("total",total)):
                sums[key]+=float(value.detach())*len(batch)
            count+=len(batch)
        row={"epoch":epoch,**{f"{key}_loss":value/count for key,value in sums.items()},
             "learning_rate":optimizer.param_groups[0]["lr"],"updates":updates,
             "elapsed_seconds":time.perf_counter()-started}
        rows.append(row)
        if epoch==1 or epoch%25==0:
            print(json.dumps(row),flush=True)
        if epoch%args.save_every==0 or epoch==args.epochs:
            torch.save(checkpoint_state(diffusion,diffusion._denoise_fn),args.output/f"model_epoch_{epoch}.pt")
            torch.save(checkpoint_state(diffusion,ema_model),args.output/f"ema_model_epoch_{epoch}.pt")
            pd.DataFrame(rows).to_csv(args.output/"training_curve.csv",index=False)

    squared=0.0; initial_squared=0.0
    for name,value in diffusion._denoise_fn.state_dict().items():
        if value.is_floating_point():
            squared+=float((value.detach().cpu()-initial[name]).square().sum())
            initial_squared+=float(initial[name].square().sum())
    metadata={
        "status":"executed","dataset":args.dataset,"epochs":args.epochs,"updates":updates,
        "graph_weight":args.graph_weight,"t_window":[args.t_min,args.t_max],
        "allowed_classifications":args.allowed_classifications or "all",
        "compiled_edges":len(programs),"skipped_edges":len(skipped),
        "full_denoiser_trainable_parameters":sum(p.numel() for p in diffusion._denoise_fn.parameters()),
        "noise_schedules_frozen":True,"clean_exceptions_supervised_as_violations":False,
        "relative_parameter_displacement":math.sqrt(squared/max(initial_squared,1e-30)),
        "runtime_seconds":time.perf_counter()-started,"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device),
        "seed":args.seed,
    }
    (args.output/"training_metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)


if __name__ == "__main__":
    main()
