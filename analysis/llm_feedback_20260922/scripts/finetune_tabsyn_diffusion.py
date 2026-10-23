#!/usr/bin/env python3
"""Continue an official TabSyn diffusion checkpoint without overwriting it.

This is a deliberately small baseline-strength audit.  It keeps the released
VAE, latent normalization and denoising objective fixed, reloads an existing
diffusion state, and restarts Adam at a lower learning rate for a fixed budget.
Checkpoint selection uses training loss only; held-out metrics are never read
by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabsyn-repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    repo = args.tabsyn_repo.resolve()
    sys.path.insert(0, str(repo))
    from tabsyn.latent_utils import get_input_train
    from tabsyn.model import MLPDiffusion, Model

    seed_all(args.seed)
    device = torch.device("cuda")
    cwd = Path.cwd()
    os.chdir(repo)
    try:
        train_z, _, _, _, _ = get_input_train(
            SimpleNamespace(dataname=args.dataset, device=str(device)))
    finally:
        os.chdir(cwd)
    train_data = (train_z - train_z.mean(0)) / 2.0
    loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
        generator=torch.Generator().manual_seed(args.seed), pin_memory=True)

    denoiser = MLPDiffusion(train_data.shape[1], 1024).to(device)
    model = Model(denoise_fn=denoiser, hid_dim=train_data.shape[1]).to(device)
    state = torch.load(args.base_checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=.9, patience=max(args.patience // 5, 10))

    best_loss = float("inf")
    stale = 0
    rows: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = batch.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        mean_loss = total / count
        scheduler.step(mean_loss)
        improved = mean_loss < best_loss
        if improved:
            best_loss = mean_loss
            stale = 0
            torch.save(model.state_dict(), args.output / "model_best.pt")
        else:
            stale += 1
        row = {
            "epoch": epoch, "loss": mean_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_loss": best_loss, "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        if epoch == 1 or epoch % 25 == 0:
            print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break

    torch.save(model.state_dict(), args.output / "model_final.pt")
    pd.DataFrame(rows).to_csv(args.output / "training_curve.csv", index=False)
    metadata = {
        "status": "executed", "method": "official TabSyn objective continuation",
        "dataset": args.dataset, "base_checkpoint": str(args.base_checkpoint),
        "selection": "minimum continuation training loss; no held-out metric",
        "requested_epochs": args.epochs, "completed_epochs": len(rows),
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "seed": args.seed, "best_loss": best_loss,
        "runtime_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(device),
    }
    (args.output / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
