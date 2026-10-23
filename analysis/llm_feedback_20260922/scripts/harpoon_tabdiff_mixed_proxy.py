#!/usr/bin/env python3
"""Harpoon-style mixed-type guidance on the same pretrained TabDiff backbone.

This is a rerun *variant*, not an official Harpoon reproduction.  It ports the
two core ideas used by Harpoon's general-constraint sampler: evaluate a
differentiable loss on the predicted clean state and apply its response at
every reverse step.  Numeric lower bounds use a squared-hinge gradient with
respect to the current noisy state.  Because TabDiff uses absorbing discrete
diffusion rather than continuous one-hot categories, categorical equality is
implemented by an analogous target-logit energy before the discrete update.
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
from tabdiff_probe import preprocess_dataset


def encoded_numeric_bound(train_data, info: dict, column: str, value: float) -> tuple[int, float]:
    names = [info["column_names"][index] for index in info["num_col_idx"]]
    position = names.index(column)
    raw = train_data.num_inverse(train_data.X[:1, : train_data.d_numerical].cpu().numpy())
    raw[0, position] = value
    encoded = train_data.num_transform.transform(raw)
    return position, float(encoded[0, position])


def categorical_target(train_data, info: dict, column: str, value: str) -> tuple[int, int]:
    names = ([info["column_names"][index] for index in info["target_col_idx"]]
             + [info["column_names"][index] for index in info["cat_col_idx"]])
    position = names.index(column)
    categories = train_data.cat_transform.named_steps["ordinal"].categories_[position]
    normalized = [str(item).strip() for item in categories]
    return position, normalized.index(str(value).strip())


def sample_guided(diffusion, samples: int, request: str, age_position: int,
                  age_bound: float, education_position: int, education_code: int,
                  numeric_step: float, categorical_logit: float,
                  start_t: float, active_min_t: float=0.0,
                  condition_position: int | None=None, condition_code: int | None=None,
                  gate_mode: str="global") -> tuple[torch.Tensor, dict]:
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num = diffusion.num_schedule.total_noise(times)
    sigma_cat = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num); sigma_num_next[1:] = sigma_num[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat); sigma_cat_next[1:] = sigma_cat[:-1]
    if diffusion.sampler_params["stochastic_sampler"] or diffusion.sampler_params["second_order_correction"]:
        raise ValueError("proxy expects deterministic first-order sampling")

    numeric = torch.randn((samples, diffusion.num_numerical_features), device=device) * sigma_num[-1]
    categorical = diffusion._sample_masked_prior(samples, len(diffusion.num_classes))
    target_offset = int(sum(diffusion.num_classes_w_mask[:education_position]))
    gradient_steps = 0
    categorical_steps = 0
    gradient_norms: list[float] = []
    frozen_gate = None
    previous_gate = None
    topology_flips = 0
    guided_row_steps = 0

    def predicted_condition(raw_logits: torch.Tensor) -> torch.Tensor:
        if condition_position is None:
            return torch.ones(samples, dtype=torch.bool, device=device)
        chunks = torch.split(raw_logits, [int(size) + 1 for size in diffusion.num_classes], dim=-1)
        size = int(diffusion.num_classes[condition_position])
        current_code = categorical[:, condition_position]
        predicted_code = chunks[condition_position][:, :size].argmax(dim=-1)
        clean_code = torch.where(current_code.eq(size), predicted_code, current_code)
        return clean_code.eq(int(condition_code))

    for index in reversed(range(diffusion.num_timesteps)):
        current_t = float(times[index])
        active = active_min_t <= current_t <= start_t
        numeric_active = active and request in {"range", "both"}
        with torch.enable_grad() if numeric_active else torch.no_grad():
            current = numeric.detach().requires_grad_(numeric_active)
            categorical_oh = diffusion.to_one_hot(categorical).to(current.dtype)
            denoised, raw_logits = diffusion._denoise_fn(
                current.float(), categorical_oh,
                times[index].squeeze().repeat(samples),
                sigma=sigma_num[index].unsqueeze(0).repeat(samples, 1),
            )
            direction = (current - denoised) / sigma_num[index]
            numeric_next = current + (sigma_num_next[index] - sigma_num[index]) * direction

            current_gate = predicted_condition(raw_logits)
            if gate_mode == "global" or condition_position is None:
                gate = torch.ones(samples, dtype=torch.bool, device=device)
            elif gate_mode == "frozen":
                if active and frozen_gate is None:
                    frozen_gate = current_gate.detach().clone()
                gate = frozen_gate if frozen_gate is not None else current_gate
            elif gate_mode == "dynamic":
                gate = current_gate
            else:
                raise ValueError(gate_mode)
            if active:
                if previous_gate is not None:
                    topology_flips += int((gate != previous_gate).sum().detach().cpu())
                previous_gate = gate.detach().clone()
                guided_row_steps += int(gate.sum().detach().cpu())

            if numeric_active:
                loss = torch.relu(torch.as_tensor(age_bound, device=device) - denoised[:, age_position]).square()
                loss = loss * gate.to(loss.dtype)
                gradient = torch.autograd.grad(loss.sum(), current, retain_graph=False, create_graph=False)[0]
                numeric_next = numeric_next.detach() - numeric_step * gradient.detach()
                gradient_norms.append(float(gradient.norm(dim=1).mean().detach().cpu()))
                gradient_steps += 1

            guided_logits = raw_logits.detach().clone()
            if active and request in {"category", "both"}:
                guided_logits[gate, target_offset + education_code] += categorical_logit
                categorical_steps += 1
            logits = diffusion._subs_parameterization(guided_logits, categorical)
            alpha_t = torch.exp(-sigma_cat[index]).unsqueeze(0).repeat(samples, 1)
            alpha_s = torch.exp(-sigma_cat_next[index]).unsqueeze(0).repeat(samples, 1)
            categorical_next, _ = diffusion._mdlm_update(logits, categorical, alpha_t, alpha_s)

        numeric = numeric_next.detach()
        categorical = categorical_next.detach()

    if not torch.all(categorical < diffusion.mask_index):
        raise RuntimeError("categorical mask remained at final step")
    encoded = torch.cat([numeric, categorical], dim=1).cpu()
    trace = {
        "gradient_steps": gradient_steps,
        "categorical_guidance_steps": categorical_steps,
        "mean_gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
        "numeric_step": numeric_step,
        "categorical_logit": categorical_logit,
        "start_t": start_t,
        "active_min_t": active_min_t,
        "gate_mode": gate_mode,
        "topology_flips": topology_flips,
        "guided_row_steps": guided_row_steps,
    }
    return encoded, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", choices=["range", "category", "both"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261376)
    parser.add_argument("--start-t", type=float, default=1.0)
    parser.add_argument("--active-min-t", type=float, default=0.0)
    parser.add_argument("--condition-column", default="",
                        help="Optional categorical antecedent used to gate the per-step energy.")
    parser.add_argument("--condition-value", default="")
    parser.add_argument("--gate-mode", choices=["global", "dynamic", "frozen"], default="global")
    parser.add_argument("--strengths", default="0.05,1:0.1,2:0.2,4:0.4,8",
                        help="Colon-separated NUMERIC_STEP,CATEGORICAL_LOGIT pairs.")
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
    age_position, age_bound = encoded_numeric_bound(train_data, info, "AGE", 50.0)
    education_position, education_code = categorical_target(train_data, info, "EDUCATION", "3")
    condition_position = condition_code = None
    if args.condition_column:
        condition_position, condition_code = categorical_target(
            train_data, info, args.condition_column, args.condition_value
        )

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
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, model, None,
        **diffusion_params, device=torch.device("cuda"),
    ).cuda()
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()

    common_seed = args.seed + 100000
    seed_all(common_seed)
    with torch.no_grad():
        unconditional = decode(train_data, diffusion.sample(args.samples), info)
    unconditional.to_csv(sample_dir / "unconditional.csv", index=False)

    rows = []
    for ordinal, item in enumerate(args.strengths.split(":")):
        numeric_step, categorical_logit = (float(value) for value in item.split(","))
        seed_all(common_seed)
        tick = time.perf_counter()
        encoded, trace = sample_guided(
            diffusion, args.samples, args.request, age_position, age_bound,
            education_position, education_code, numeric_step, categorical_logit,
            args.start_t, args.active_min_t, condition_position, condition_code,
            args.gate_mode,
        )
        frame = decode(train_data, encoded, info)
        label = f"proxy_n{numeric_step:g}_c{categorical_logit:g}"
        frame.to_csv(sample_dir / f"{label}.csv", index=False)
        trace.update(label=label, runtime_seconds=time.perf_counter() - tick)
        rows.append(trace)
        (args.output / f"trace_{label}.json").write_text(json.dumps(trace, indent=2))

    pd.DataFrame(rows).to_csv(args.output / "guidance_trace.csv", index=False)
    metadata = {
        "label": "Harpoon-style mixed-type TabDiff proxy; not an official Harpoon reproduction",
        "request": args.request,
        "samples": args.samples,
        "steps": args.steps,
        "seed": args.seed,
        "start_t": args.start_t,
        "active_min_t": args.active_min_t,
        "condition_column": args.condition_column or None,
        "condition_value": args.condition_value if args.condition_column else None,
        "gate_mode": args.gate_mode,
        "age_encoded_bound": age_bound,
        "education_code": education_code,
        "numeric_response": "squared-hinge gradient on predicted-clean state",
        "categorical_response": "target-logit energy before absorbing discrete update",
        "runtime_seconds": time.perf_counter() - started,
        "api_calls": 0,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
