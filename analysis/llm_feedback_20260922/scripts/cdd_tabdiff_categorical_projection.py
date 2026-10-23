#!/usr/bin/env python3
"""Matched CDD-style categorical projection on a pretrained TabDiff sampler.

This is a domain adaptation, not an official reproduction of CDD.  CDD was
evaluated on discrete sequences, whereas TabDiff uses one absorbing diffusion
chain per categorical table column.  We preserve its two relevant ingredients:
(1) KL-nearest projection of the denoiser distribution onto an argmax-feasible
set, and (2) insertion before every selected reverse update.  For the
constrained column we use greedy feasible decoding, matching CDD's decoded
argmax constraint; all other columns retain TabDiff's stochastic transition.
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
from harpoon_tabdiff_mixed_proxy import categorical_target
from tabdiff_probe import preprocess_dataset


def kl_argmax_projection(probabilities: torch.Tensor, target: int) -> torch.Tensor:
    """Project q to argmax(p)=target under D_KL(q || p).

    The exact I-projection pools the target with every category whose mass is
    above the pooled mean.  Unpooled coordinates remain unchanged.
    """
    result = probabilities.clone()
    active = torch.zeros_like(probabilities, dtype=torch.bool)
    active[:, target] = True
    while True:
        pooled = (probabilities * active).sum(-1, keepdim=True) / active.sum(-1, keepdim=True)
        new_active = active | (probabilities > pooled)
        if torch.equal(new_active, active):
            break
        active = new_active
    pooled = (probabilities * active).sum(-1, keepdim=True) / active.sum(-1, keepdim=True)
    result = torch.where(active, pooled, result)
    return result / result.sum(-1, keepdim=True)


def sample_projected(diffusion, samples: int, column_position: int, target_code: int,
                     start_t: float, active_min_t: float) -> tuple[torch.Tensor, dict]:
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num = diffusion.num_schedule.total_noise(times)
    sigma_cat = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num); sigma_num_next[1:] = sigma_num[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat); sigma_cat_next[1:] = sigma_cat[:-1]
    if diffusion.sampler_params["stochastic_sampler"] or diffusion.sampler_params["second_order_correction"]:
        raise ValueError("matched adapter expects deterministic first-order numeric sampling")

    numeric = torch.randn((samples, diffusion.num_numerical_features), device=device) * sigma_num[-1]
    categorical = diffusion._sample_masked_prior(samples, len(diffusion.num_classes))
    offset = int(sum(diffusion.num_classes_w_mask[:column_position]))
    cardinality = int(diffusion.num_classes[column_position])
    projection_steps = 0
    reopened_rows = 0
    kl_values: list[float] = []

    for index in reversed(range(diffusion.num_timesteps)):
        current_t = float(times[index])
        active = active_min_t <= current_t <= start_t
        if active:
            wrong_fixed = categorical[:, column_position].ne(diffusion.mask_index[column_position]) & categorical[:, column_position].ne(target_code)
            reopened_rows += int(wrong_fixed.sum().cpu())
            categorical[wrong_fixed, column_position] = diffusion.mask_index[column_position]

        with torch.no_grad():
            categorical_oh = diffusion.to_one_hot(categorical).to(numeric.dtype)
            denoised, raw_logits = diffusion._denoise_fn(
                numeric.float(), categorical_oh,
                times[index].squeeze().repeat(samples),
                sigma=sigma_num[index].unsqueeze(0).repeat(samples, 1),
            )
            direction = (numeric - denoised) / sigma_num[index]
            numeric_next = numeric + (sigma_num_next[index] - sigma_num[index]) * direction

            projected_logits = raw_logits.clone()
            if active:
                target_slice = slice(offset, offset + cardinality)
                q = torch.softmax(raw_logits[:, target_slice], dim=-1).clamp_min(1e-12)
                p = kl_argmax_projection(q, target_code).clamp_min(1e-12)
                kl_values.append(float((q * (q.log() - p.log())).sum(-1).mean().cpu()))
                projected_logits[:, target_slice] = p.log()
                projection_steps += 1

            logits = diffusion._subs_parameterization(projected_logits, categorical)
            alpha_t = torch.exp(-sigma_cat[index]).unsqueeze(0).repeat(samples, 1)
            alpha_s = torch.exp(-sigma_cat_next[index]).unsqueeze(0).repeat(samples, 1)
            categorical_next, q_xs = diffusion._mdlm_update(logits, categorical, alpha_t, alpha_s)

            if active:
                masked = categorical[:, column_position].eq(diffusion.mask_index[column_position])
                if masked.any():
                    col_q = q_xs[:, column_position]
                    mask_idx = int(diffusion.mask_index[column_position])
                    mask_prob = col_q[:, mask_idx]
                    total = col_q.sum(-1).clamp_min(1e-12)
                    stay_masked = torch.rand(samples, device=device) < (mask_prob / total)
                    constrained = torch.full_like(categorical_next[:, column_position], target_code)
                    constrained[stay_masked] = mask_idx
                    categorical_next[masked, column_position] = constrained[masked]

        numeric = numeric_next.detach()
        categorical = categorical_next.detach()

    if not torch.all(categorical < diffusion.mask_index):
        raise RuntimeError("categorical mask remained at final step")
    encoded = torch.cat([numeric, categorical], dim=1).cpu()
    return encoded, {
        "projection_steps": projection_steps,
        "reopened_rows": reopened_rows,
        "mean_projection_kl": float(np.mean(kl_values)) if kl_values else 0.0,
        "start_t": start_t,
        "active_min_t": active_min_t,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--windows", default="full,1,0:middle,.75,.25")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sample_dir = args.output / "samples"; sample_dir.mkdir()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    started = time.perf_counter()
    info = json.loads((args.repo / "data/default/info.json").read_text())
    train_data, _ = preprocess_dataset(args.repo, "default")
    position, target = categorical_target(train_data, info, "EDUCATION", "3")
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["denoise_fn"], strict=True)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column",
                            cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, model, None,
        **diffusion_params, device=torch.device("cuda")).cuda()
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()

    common_seed = args.seed + 100000
    seed_all(common_seed)
    with torch.no_grad():
        decode(train_data, diffusion.sample(args.samples), info).to_csv(sample_dir / "unconditional.csv", index=False)

    traces = []
    for item in args.windows.split(":"):
        label, start_t, active_min_t = item.split(",")
        seed_all(common_seed)
        tick = time.perf_counter()
        encoded, trace = sample_projected(diffusion, args.samples, position, target, float(start_t), float(active_min_t))
        decode(train_data, encoded, info).to_csv(sample_dir / f"cdd_{label}.csv", index=False)
        trace.update(label=f"cdd_{label}", runtime_seconds=time.perf_counter() - tick)
        traces.append(trace)
    pd.DataFrame(traces).to_csv(args.output / "projection_trace.csv", index=False)
    metadata = {
        "label": "CDD-style matched TabDiff adaptation; not an official CDD reproduction",
        "source_method": "Constrained Discrete Diffusion, NeurIPS 2025",
        "constraint": "EDUCATION = 3",
        "samples": args.samples, "steps": args.steps, "seed": args.seed,
        "projection": "exact D_KL(q||p) argmax-feasible I-projection",
        "decoder": "greedy feasible transition for constrained coordinate; stochastic TabDiff elsewhere",
        "runtime_seconds": time.perf_counter() - started, "api_calls": 0,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(traces).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
