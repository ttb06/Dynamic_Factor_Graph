#!/usr/bin/env python3
"""Sample an ordinary TabDiff-compatible checkpoint on any prepared dataset."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import torch

from conditioned_tabdiff_adapter import decode, seed_all
from tabdiff_probe import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    seed_all(args.seed)
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    train_data, _ = preprocess_dataset(args.repo, args.dataset)
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["denoise_fn"], strict=True)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(
        num_timesteps=args.steps,
        scheduler="power_mean_per_column",
        cat_scheduler="log_linear_per_column",
    )
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(
        stochastic_sampler=False,
        second_order_correction=False,
    )
    device = torch.device("cuda")
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, model, None,
        **diffusion_params, device=device,
    ).to(device)
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()
    with torch.no_grad():
        encoded = diffusion.sample(args.samples)
    frame = decode(train_data, encoded, info)
    frame.to_csv(args.output, index=False)
    print(json.dumps({
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "samples": len(frame),
        "steps": args.steps,
        "seed": args.seed,
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
