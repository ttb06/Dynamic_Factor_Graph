#!/usr/bin/env python3
"""Train a bounded Adult TabDiff pilot without upstream's expensive periodic reports."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from tabdiff_probe import preprocess_dataset


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_ema(target, source, rate: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.detach().mul_(rate).add_(source_param.detach(), alpha=1 - rate)


def checkpoint_state(diffusion, use_ema=False, ema_model=None, ema_num=None, ema_cat=None):
    return {
        "denoise_fn": (ema_model if use_ema else diffusion._denoise_fn).state_dict(),
        "num_schedule": (ema_num if use_ema else diffusion.num_schedule).state_dict(),
        "cat_schedule": (ema_cat if use_ema else diffusion.cat_schedule).state_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=["adult", "default", "shoppers", "magic", "diabetes", "news"],
        default="adult",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--num-timesteps", type=int, default=30,
        help="Diffusion discretization used in training; released TabDiff uses 50.",
    )
    parser.add_argument(
        "--regression-target-mode", choices=["categorical", "numerical"], default="categorical",
        help="Use numerical to match released TabDiff preprocessing for regression datasets.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))

    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    if not torch.cuda.is_available():
        raise RuntimeError("Pilot training is submitted as a GPU job, but torch.cuda.is_available() is false")
    device = torch.device("cuda")
    seed_all(args.seed)
    train_data, _ = preprocess_dataset(
        args.repo, args.dataset, regression_target_mode=args.regression_target_mode)
    loader = DataLoader(
        TensorDataset(train_data.X),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    with args.base_config.open("rb") as handle:
        config = pickle.load(handle)
    config = copy.deepcopy(config)
    config["train"]["main"]["steps"] = args.epochs
    config["train"]["main"]["batch_size"] = args.batch_size
    config["train"]["main"]["lr"] = args.learning_rate
    config["train"]["main"]["check_val_every"] = args.save_every
    config["diffusion_params"]["num_timesteps"] = args.num_timesteps
    config["unimodmlp_params"]["d_numerical"] = train_data.d_numerical
    config["unimodmlp_params"]["categories"] = (train_data.categories + 1).tolist()
    config["pilot_scope"] = {
        "purpose": "bounded mechanism pilot, not paper reproduction",
        "dataset": args.dataset,
        "seed": args.seed,
        "regression_target_mode": args.regression_target_mode,
        "num_timesteps": args.num_timesteps,
        "no_periodic_generation": True,
    }
    with (args.output / "config.pkl").open("wb") as handle:
        pickle.dump(config, handle)

    backbone = UniModMLP(**config["unimodmlp_params"])
    model = Model(backbone, **config["diffusion_params"]["edm_params"]).to(device)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params["scheduler"] = "power_mean_per_column"
    diffusion_params["cat_scheduler"] = "log_linear_per_column"
    diffusion = UnifiedCtimeDiffusion(
        num_classes=train_data.categories,
        num_numerical_features=train_data.d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **diffusion_params,
        device=device,
    ).to(device)
    ema_model = copy.deepcopy(diffusion._denoise_fn).eval()
    ema_num = copy.deepcopy(diffusion.num_schedule).eval()
    ema_cat = copy.deepcopy(diffusion.cat_schedule).eval()
    for module in (ema_model, ema_num, ema_cat):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=50)
    log_rows = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        c_weight = 1.0 - (epoch - 1) / args.epochs
        d_sum, c_sum, count = 0.0, 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            dloss, closs = diffusion.mixed_loss(batch)
            loss = dloss + c_weight * closs
            loss.backward()
            optimizer.step()
            d_sum += float(dloss.detach()) * len(batch)
            c_sum += float(closs.detach()) * len(batch)
            count += len(batch)
        d_mean, c_mean = d_sum / count, c_sum / count
        scheduler.step(d_mean + c_mean)
        update_ema(ema_model, diffusion._denoise_fn, 0.997)
        update_ema(ema_num, diffusion.num_schedule, 0.997)
        update_ema(ema_cat, diffusion.cat_schedule, 0.997)
        row = {
            "epoch": epoch,
            "discrete_loss": d_mean,
            "continuous_loss": c_mean,
            "total_unweighted_loss": d_mean + c_mean,
            "categorical_weight": c_weight,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - start,
        }
        log_rows.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(checkpoint_state(diffusion), args.output / f"model_epoch_{epoch}.pt")
            torch.save(
                checkpoint_state(diffusion, True, ema_model, ema_num, ema_cat),
                args.output / f"ema_model_epoch_{epoch}.pt",
            )
            pd.DataFrame(log_rows).to_csv(args.output / "training_curve.csv", index=False)

    metadata = {
        "status": "executed",
        "scope": "bounded pilot, not TabDiff paper reproduction",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "updates": args.epochs * len(loader),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "regression_target_mode": args.regression_target_mode,
        "num_timesteps": args.num_timesteps,
        "d_numerical": train_data.d_numerical,
        "categories": (train_data.categories + 1).tolist(),
        "runtime_seconds": time.perf_counter() - start,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    (args.output / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
