#!/usr/bin/env python3
"""Port the typed row-edge-time graph solver to TabSyn's latent sampler.

The graph is discovered/verified offline.  During sampling, every predictor
preview updates row-specific edge readability; only stable row-edge pairs in
the selected noise interval produce a typed data-space action.  The action is
re-encoded and injected as a bounded latent displacement.  No API or learned
critic is called online.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dense_numeric_factor_messages import DenseNumericFactors
from tabsyn_dynamic_graph_guidance import TabSynCodec, seed_all


def build_engine(codec: TabSynCodec, train: pd.DataFrame, edges: list[dict]):
    """Construct only the deterministic typed-program part of the engine."""
    engine = DenseNumericFactors.__new__(DenseNumericFactors)
    engine.train_data = None
    engine.train = train
    engine.edges = edges
    numeric_names = [codec.column_names[index] for index in codec.num_idx]
    target_names = [codec.column_names[index] for index in codec.target_idx]
    engine.numeric_names = numeric_names
    engine.model_numeric_names = (
        target_names + numeric_names if codec.info["task_type"] == "regression"
        else numeric_names)
    engine.numeric_set = set(numeric_names)
    engine.num_pos = {name: index for index, name in enumerate(engine.model_numeric_names)}
    engine.scales = {
        name: max(float(pd.to_numeric(train[name], errors="coerce").std()), 1e-6)
        for name in numeric_names
    }
    engine.models = {}
    return engine


def interval_weight(q: float, lower: float, upper: float, schedule: str) -> float:
    if q < lower or q > upper:
        return 0.0
    if schedule == "flat":
        return 1.0
    midpoint = (lower + upper) / 2.0
    half = max((upper - lower) / 2.0, 1e-12)
    return max(0.0, 1.0 - abs(q - midpoint) / half)


def stable_masks(engine, frame: pd.DataFrame,
                 previous: dict[str, np.ndarray],
                 streaks: dict[str, np.ndarray], threshold: float,
                 persistence: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    current: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for edge in engine.edges:
        edge_id = edge["candidate_id"]
        item = engine.program_signal(frame, edge)
        if item is None:
            continue
        signal, scale = item
        current[edge_id] = signal
        one_step = engine.stable_signal_mask(
            signal, previous.get(edge_id), threshold, "rank", scale)
        old = streaks.get(edge_id, np.zeros(len(frame), dtype=np.int16))
        new = np.where(one_step, old + 1, 0).astype(np.int16)
        streaks[edge_id] = new
        masks[edge_id] = new >= persistence
    return current, masks


@torch.no_grad()
def inject(clean: torch.Tensor, codec: TabSynCodec, engine,
           row_masks: dict[str, np.ndarray], strength: float,
           trust_radius: float) -> tuple[torch.Tensor, dict]:
    preview = codec.decode(clean)
    corrected, changed, events = engine.project_violations(
        preview, strength, edge_row_masks=row_masks)
    active_rows = changed.any(axis=1)
    if not active_rows.any():
        return clean, {"events": int(events), "rows": 0, "movement": 0.0}
    before = codec.encode(preview)
    after = codec.encode(corrected)
    delta = after - before
    norms = delta.norm(dim=1, keepdim=True)
    if trust_radius > 0:
        delta = delta * torch.clamp(trust_radius / norms.clamp_min(1e-12), max=1.0)
    movement = float(delta.norm(dim=1).sum().cpu())
    return clean + delta, {
        "events": int(events), "rows": int(active_rows.sum()), "movement": movement}


@torch.no_grad()
def sample(net, codec: TabSynCodec, engine, samples: int, steps: int, seed: int,
           q_min: float, q_max: float, schedule_name: str,
           strength: float, trust_radius: float, threshold: float,
           persistence: int, guidance_stage: str, guided: bool):
    from tabsyn.diffusion_utils import (
        SIGMA_MAX, SIGMA_MIN, S_churn, S_max, S_min, S_noise, rho)

    seed_all(seed)
    latents = torch.randn((samples, codec.train_z.shape[1]), device=codec.device)
    indices = torch.arange(steps, dtype=torch.float32, device=codec.device)
    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)
    sigmas = (sigma_max ** (1 / rho) + indices / (steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigmas = torch.cat([net.round_sigma(sigmas), torch.zeros_like(sigmas[:1])])
    log_sigma = sigmas[:-1].clamp_min(1e-12).log()
    q_values = (log_sigma - log_sigma.min()) / (log_sigma.max() - log_sigma.min())
    x_next = latents.float() * sigmas[0]
    previous: dict[str, np.ndarray] = {}
    streaks: dict[str, np.ndarray] = {}
    totals = defaultdict(float)
    trace: list[dict] = []

    for index, (t_cur, t_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        x_cur = x_next
        gamma = min(S_churn / steps, math.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat.square() - t_cur.square()).sqrt() * S_noise * torch.randn_like(x_cur)
        denoised = net(x_hat, t_hat).float()
        q = float(q_values[index])
        preview = codec.decode(denoised)
        current, masks = stable_masks(
            engine, preview, previous, streaks, threshold, persistence)
        weight = interval_weight(q, q_min, q_max, schedule_name)
        step = {
            "step": index, "sigma": float(t_cur), "log_sigma_quantile": q,
            "interval_weight": weight,
            "active_row_edge_fraction": float(
                sum(int(value.sum()) for value in masks.values())
                / max(samples * len(masks), 1)),
            "predictor_events": 0, "corrector_events": 0,
            "rows": 0, "latent_movement": 0.0,
        }
        if (guided and weight > 0 and guidance_stage in {"predictor", "both"}):
            denoised, diagnostic = inject(
                denoised, codec, engine, masks, strength * weight, trust_radius)
            step["predictor_events"] = diagnostic["events"]
            step["rows"] += diagnostic["rows"]
            step["latent_movement"] += diagnostic["movement"]

        derivative = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * derivative
        if index < steps - 1:
            denoised_next = net(x_next, t_next).float()
            q_next = float(q_values[index + 1])
            next_weight = interval_weight(q_next, q_min, q_max, schedule_name)
            if (guided and next_weight > 0
                    and guidance_stage in {"corrector", "both"}):
                denoised_next, diagnostic = inject(
                    denoised_next, codec, engine, masks,
                    strength * next_weight, trust_radius)
                step["corrector_events"] = diagnostic["events"]
                step["rows"] += diagnostic["rows"]
                step["latent_movement"] += diagnostic["movement"]
            derivative_next = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (
                .5 * derivative + .5 * derivative_next)
        previous = current
        totals["events"] += step["predictor_events"] + step["corrector_events"]
        totals["row_events"] += step["rows"]
        totals["latent_movement"] += step["latent_movement"]
        trace.append(step)
    return codec.decode(x_next), {**dict(totals), "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabsyn-repo", type=Path, required=True)
    parser.add_argument("--dataset", default="news")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--q-min", type=float, default=.10)
    parser.add_argument("--q-max", type=float, default=.40)
    parser.add_argument("--schedule", choices=["flat", "middle_peak"], default="middle_peak")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--trust-radius", type=float, default=.5)
    parser.add_argument("--stability-threshold", type=float, default=.001)
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--guidance-stage", choices=["predictor", "corrector", "both"], default="both")
    parser.add_argument("--family-strengths", default="")
    parser.add_argument("--family-transports", default="")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    codec = TabSynCodec(args.tabsyn_repo.resolve(), args.dataset, device)
    train = pd.read_csv(args.tabsyn_repo / "synthetic" / args.dataset / "real.csv") \
        if (args.tabsyn_repo / "synthetic" / args.dataset / "real.csv").exists() \
        else pd.read_csv(Path.cwd() / "synthetic" / args.dataset / "real.csv")
    edges = copy.deepcopy(json.loads(args.graph.read_text()))
    family_strengths = {}
    for item in filter(None, args.family_strengths.split(",")):
        family, value = item.split("=", 1); family_strengths[family] = float(value)
    family_transports = {}
    for item in filter(None, args.family_transports.split(",")):
        family, value = item.split("=", 1); family_transports[family] = value
    for edge in edges:
        edge["guidance_weight"] = family_strengths.get(edge.get("relation_family"), 1.0)
        if edge.get("program") and edge.get("relation_family") in family_transports:
            edge["program"]["transport"] = family_transports[edge["relation_family"]]
    engine = build_engine(codec, train, edges)

    from tabsyn.model import MLPDiffusion, Model
    denoiser = MLPDiffusion(codec.train_z.shape[1], 1024).to(device)
    model = Model(denoise_fn=denoiser, hid_dim=codec.train_z.shape[1]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    net = model.denoise_fn_D.eval()
    common = (net, codec, engine, args.samples, args.steps, args.seed,
              args.q_min, args.q_max, args.schedule, args.strength,
              args.trust_radius, args.stability_threshold, args.persistence,
              args.guidance_stage)
    baseline, baseline_trace = sample(*common, guided=False)
    guided, guided_trace = sample(*common, guided=True)
    baseline.to_csv(args.output / "tabsyn_baseline.csv", index=False)
    guided.to_csv(args.output / "tabsyn_typed_dynamic.csv", index=False)
    (args.output / "trace.json").write_text(json.dumps({
        "baseline": baseline_trace, "guided": guided_trace}, indent=2))
    metadata = {
        **vars(args), "checkpoint": str(args.checkpoint), "graph": str(args.graph),
        "output": str(args.output), "edges": len(edges),
        "mechanism": "offline typed graph; dynamic row-edge state; bounded TabSyn latent injection",
        "api_calls_during_sampling": 0,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(json.dumps({
        "output": str(args.output), "edges": len(edges), **guided_trace},
        default=lambda value: "<trace>" if isinstance(value, list) else value)[:4000], flush=True)


if __name__ == "__main__":
    main()
