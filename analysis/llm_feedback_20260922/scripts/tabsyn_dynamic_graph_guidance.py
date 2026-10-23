#!/usr/bin/env python3
"""Training-free dynamic graph guidance for the official TabSyn sampler.

This is a backbone-portability experiment, not a reimplementation of TabSyn.
At selected reverse steps we decode TabSyn's clean latent prediction, apply a
small set of independently checkable Shoppers relations, re-encode both the
original and corrected previews, and inject only their latent difference.  A
row-level trust radius prevents the VAE round trip from becoming an implicit
full-record repair.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TabSynCodec:
    def __init__(self, tabsyn_repo: Path, dataset: str, device: torch.device):
        self.repo = tabsyn_repo
        self.dataset = dataset
        self.device = device
        sys.path.insert(0, str(tabsyn_repo))
        from src import Transformations
        from tabsyn.vae.model import Encoder_model
        from utils_train import make_dataset, preprocess
        from tabsyn.latent_utils import get_input_generate

        data_dir = tabsyn_repo / "data" / dataset
        self.info = json.loads((data_dir / "info.json").read_text())
        args = SimpleNamespace(dataname=dataset, device=str(device))
        cwd = Path.cwd()
        os.chdir(tabsyn_repo)
        try:
            (self.train_z, _, _, self.diffusion_ckpt, loaded_info,
             self.num_inverse, self.cat_inverse) = get_input_generate(args)
            transformations = Transformations(
                normalization="quantile", num_nan_policy="mean",
                cat_nan_policy=None, cat_min_frequency=None,
                cat_encoding=None, y_policy="default")
            transformed = make_dataset(
                str(data_dir), transformations, self.info["task_type"],
                change_val=False, concat=True)
            x_num, x_cat, categories, d_numerical = preprocess(
                str(data_dir), task_type=self.info["task_type"])
        finally:
            os.chdir(cwd)

        self.num_transform = transformed.num_transform
        self.cat_transform = transformed.cat_transform
        self.decoder = loaded_info["pre_decoder"].to(device).eval()
        self.token_dim = int(loaded_info["token_dim"])
        self.encoder = Encoder_model(
            2, d_numerical, categories, 4, n_head=1, factor=32)
        encoder_path = tabsyn_repo / "tabsyn" / "vae" / "ckpt" / dataset / "encoder.pt"
        self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        self.encoder = self.encoder.to(device).eval()
        self.mean = self.train_z.mean(0).to(device)
        self.column_names = list(self.info["column_names"])
        self.num_idx = list(self.info["num_col_idx"])
        self.cat_idx = list(self.info["cat_col_idx"])
        self.target_idx = list(self.info["target_col_idx"])

    @torch.no_grad()
    def decode(self, normalized_latent: torch.Tensor) -> pd.DataFrame:
        raw_latent = normalized_latent * 2.0 + self.mean
        tokens = raw_latent.reshape(len(raw_latent), -1, self.token_dim)
        x_num_transformed, cat_logits = self.decoder(tokens)
        cat_codes = torch.stack([value.argmax(dim=-1) for value in cat_logits], dim=1)
        numeric = self.num_inverse(x_num_transformed.float().cpu().numpy())
        categorical = self.cat_inverse(cat_codes.cpu().numpy())

        if self.info["task_type"] == "regression":
            n_target = len(self.target_idx)
            target = numeric[:, :n_target]
            numeric = numeric[:, n_target:]
        else:
            n_target = len(self.target_idx)
            target = categorical[:, :n_target]
            categorical = categorical[:, n_target:]

        frame = pd.DataFrame(index=np.arange(len(normalized_latent)), columns=self.column_names)
        for position, column_index in enumerate(self.num_idx):
            frame.iloc[:, column_index] = numeric[:, position]
        for position, column_index in enumerate(self.cat_idx):
            frame.iloc[:, column_index] = categorical[:, position]
        for position, column_index in enumerate(self.target_idx):
            frame.iloc[:, column_index] = target[:, position]
        for column_index in self.num_idx:
            name = self.column_names[column_index]
            frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
        return frame

    @torch.no_grad()
    def encode(self, frame: pd.DataFrame) -> torch.Tensor:
        numeric_raw = frame.iloc[:, self.num_idx].to_numpy(dtype=float)
        categorical_features = frame.iloc[:, self.cat_idx].to_numpy(dtype=object)
        target = frame.iloc[:, self.target_idx].to_numpy(dtype=object)
        if self.info["task_type"] == "regression":
            numeric_raw = np.concatenate([target.astype(float), numeric_raw], axis=1)
            categorical_raw = categorical_features
        else:
            categorical_raw = np.concatenate([target, categorical_features], axis=1)
        numeric = self.num_transform.transform(numeric_raw)
        categorical = self.cat_transform.transform(categorical_raw)
        numeric_tensor = torch.as_tensor(numeric, dtype=torch.float32, device=self.device)
        categorical_tensor = torch.as_tensor(
            categorical, dtype=torch.long, device=self.device)
        latent = self.encoder(numeric_tensor, categorical_tensor)[:, 1:, :]
        latent = latent.reshape(len(frame), -1)
        return (latent - self.mean) / 2.0


def shoppers_projection(frame: pd.DataFrame,
                        previous: dict[str, np.ndarray] | None,
                        require_persistence: bool,
                        active_override: dict[str, np.ndarray] | None = None,
                        ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    """Minimum raw-space action for three documented Shoppers relations."""
    result = frame.copy()
    current: dict[str, np.ndarray] = {}
    details: dict[str, int] = {}
    changed = np.zeros(len(frame), dtype=bool)

    for count, duration, label in (
        ("Administrative", "Administrative_Duration", "administrative_zero"),
        ("Informational", "Informational_Duration", "informational_zero"),
    ):
        count_value = pd.to_numeric(frame[count], errors="coerce").fillna(0).to_numpy(float)
        duration_value = pd.to_numeric(frame[duration], errors="coerce").fillna(0).to_numpy(float)
        bad = (count_value < 0.5) & (duration_value > 1e-8)
        current[label] = bad
        active = bad if not require_persistence or previous is None else bad & previous.get(label, bad)
        if active_override is not None:
            active &= active_override.get(label, np.zeros(len(frame), dtype=bool))
        result.loc[active, duration] = 0.0
        changed |= active
        details[label] = int(active.sum())

    bounce = pd.to_numeric(frame["BounceRates"], errors="coerce").fillna(0).to_numpy(float)
    exit_rate = pd.to_numeric(frame["ExitRates"], errors="coerce").fillna(0).to_numpy(float)
    bad = bounce > exit_rate + 1e-8
    current["bounce_le_exit"] = bad
    active = bad if not require_persistence or previous is None else bad & previous.get("bounce_le_exit", bad)
    if active_override is not None:
        active &= active_override.get("bounce_le_exit", np.zeros(len(frame), dtype=bool))
    midpoint = (bounce + exit_rate) / 2.0
    result.loc[active, "BounceRates"] = midpoint[active]
    result.loc[active, "ExitRates"] = midpoint[active]
    changed |= active
    details["bounce_le_exit"] = int(active.sum())
    details["unique_rows"] = int(changed.sum())
    return result, current, details


def shoppers_violation_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return typed row-edge states without applying an action."""
    masks: dict[str, np.ndarray] = {}
    for count, duration, label in (
        ("Administrative", "Administrative_Duration", "administrative_zero"),
        ("Informational", "Informational_Duration", "informational_zero"),
    ):
        count_value = pd.to_numeric(frame[count], errors="coerce").fillna(0).to_numpy(float)
        duration_value = pd.to_numeric(frame[duration], errors="coerce").fillna(0).to_numpy(float)
        masks[label] = (count_value < 0.5) & (duration_value > 1e-8)
    bounce = pd.to_numeric(frame["BounceRates"], errors="coerce").fillna(0).to_numpy(float)
    exit_rate = pd.to_numeric(frame["ExitRates"], errors="coerce").fillna(0).to_numpy(float)
    masks["bounce_le_exit"] = bounce > exit_rate + 1e-8
    return masks


def shoppers_edge_signals(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Endpoint signals used to decide when a typed edge is readable.

    Violation membership is intentionally not used: an early violation can
    disappear naturally and gives a brittle route.  Endpoint ranks instead
    ask whether the row's semantic state has stabilized across previews.
    """
    def values(*columns: str) -> np.ndarray:
        return frame[list(columns)].apply(
            pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)

    return {
        "administrative_zero": values("Administrative", "Administrative_Duration"),
        "informational_zero": values("Informational", "Informational_Duration"),
        "bounce_le_exit": values("BounceRates", "ExitRates"),
    }


def stable_rank_masks(current: dict[str, np.ndarray],
                      previous: dict[str, np.ndarray] | None,
                      threshold: float) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for edge_id, signal in current.items():
        if previous is None or edge_id not in previous:
            masks[edge_id] = np.zeros(len(signal), dtype=bool)
            continue
        prior = previous[edge_id]
        stable = np.ones(len(signal), dtype=bool)
        for column in range(signal.shape[1]):
            current_order = np.argsort(signal[:, column], kind="mergesort")
            prior_order = np.argsort(prior[:, column], kind="mergesort")
            current_rank = np.empty(len(signal), dtype=float)
            prior_rank = np.empty(len(signal), dtype=float)
            current_rank[current_order] = np.arange(len(signal))
            prior_rank[prior_order] = np.arange(len(signal))
            stable &= np.abs(current_rank - prior_rank) / max(len(signal) - 1, 1) <= threshold
        masks[edge_id] = stable
    return masks


@torch.no_grad()
def sample_once(net, codec: TabSynCodec, samples: int, steps: int, seed: int,
                strength: float, q_min: float, q_max: float,
                trust_radius: float, require_persistence: bool,
                route_mode: str, actuation_min_q: float, actuation_max_q: float,
                stability_threshold: float,
                guidance_stage: str,
                guided: bool) -> tuple[pd.DataFrame, dict]:
    from tabsyn.diffusion_utils import SIGMA_MAX, SIGMA_MIN, S_churn, S_max, S_min, S_noise, rho

    seed_all(seed)
    latents = torch.randn((samples, codec.train_z.shape[1]), device=codec.device)
    indices = torch.arange(steps, dtype=torch.float32, device=codec.device)
    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)
    schedule = (sigma_max ** (1 / rho) + indices / (steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    schedule = torch.cat([net.round_sigma(schedule), torch.zeros_like(schedule[:1])])
    log_sigma = schedule[:-1].clamp_min(1e-12).log()
    q_values = (log_sigma - log_sigma.min()) / (log_sigma.max() - log_sigma.min())
    x_next = latents.float() * schedule[0]
    previous_masks: dict[str, np.ndarray] | None = None
    previous_signals: dict[str, np.ndarray] | None = None
    route_candidates: list[dict] = []
    routed_masks: dict[str, np.ndarray] = {}
    route_captured = False
    trace = []

    for index, (t_cur, t_next) in enumerate(zip(schedule[:-1], schedule[1:])):
        x_cur = x_next
        gamma = min(S_churn / steps, math.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat.square() - t_cur.square()).sqrt() * S_noise * torch.randn_like(x_cur)
        denoised = net(x_hat, t_hat).float()
        q = float(q_values[index])
        step = {"step": index, "sigma": float(t_cur), "log_sigma_quantile": q,
                "unique_rows": 0, "latent_movement": 0.0}

        if (guided and strength > 0 and guidance_stage in {"predictor", "both"}
                and route_mode == "fixed" and q_min <= q <= q_max):
            preview = codec.decode(denoised)
            corrected, current_masks, diagnostics = shoppers_projection(
                preview, previous_masks, require_persistence)
            previous_masks = current_masks
            if diagnostics["unique_rows"]:
                encoded_before = codec.encode(preview)
                encoded_after = codec.encode(corrected)
                delta = encoded_after - encoded_before
                norms = delta.norm(dim=1, keepdim=True)
                if trust_radius > 0:
                    delta = delta * torch.clamp(
                        trust_radius / norms.clamp_min(1e-12), max=1.0)
                denoised = denoised + strength * delta
                step["latent_movement"] = float(
                    (strength * delta).norm(dim=1).sum().cpu())
            step.update(diagnostics)

        elif guided and strength > 0 and route_mode in {"auto_knee", "auto_valley"}:
            preview = codec.decode(denoised)
            current_masks = shoppers_violation_masks(preview)
            current_signals = shoppers_edge_signals(preview)
            stable_masks = stable_rank_masks(
                current_signals, previous_signals, stability_threshold)
            stable_fraction = float(
                sum(int(mask.sum()) for mask in stable_masks.values())
                / max(samples * len(stable_masks), 1)
            )
            if not route_captured:
                route_candidates.append({
                    "q": q,
                    "masks": {edge_id: mask.copy() for edge_id, mask in stable_masks.items()},
                    "stable_violation_fraction": stable_fraction,
                })
                step["route_candidate_fraction"] = stable_fraction
                if q <= actuation_max_q and len(route_candidates) >= 3:
                    coordinates = np.asarray([row["q"] for row in route_candidates], dtype=float)
                    fractions = np.asarray(
                        [row["stable_violation_fraction"] for row in route_candidates], dtype=float)
                    delta_q = np.abs(np.diff(coordinates)).clip(min=1e-12)
                    # Some latent backbones emit nearly collapsed previews at
                    # the noisiest states, producing high but meaningless rank
                    # stability.  auto_valley discards that false-confidence
                    # prefix and fits the increasing denoising branch only.
                    # Candidate zero has no previous preview, hence every
                    # stability mask is undefined/false by construction.  It
                    # must not be treated as a real readability valley.
                    if route_mode == "auto_valley" and len(fractions) > 2:
                        start = int(np.argmin(fractions[1:])) + 1
                    else:
                        start = 0
                    suffix = fractions[start:]
                    fitted = IsotonicRegression(increasing=True).fit_transform(
                        np.arange(len(suffix), dtype=float), suffix)
                    slopes = np.diff(fitted) / delta_q[start:]
                    slope_index = int(np.argmax(slopes))
                    selected_index = start + slope_index + 1
                    selected = route_candidates[selected_index]
                    routed_masks = selected["masks"]
                    route_captured = True
                    step.update({
                        "route_captured": True,
                        "auto_route_q": float(selected["q"]),
                        "auto_route_max_slope": float(slopes[slope_index]),
                        "auto_route_valley_q": float(route_candidates[start]["q"]),
                        "routed_row_edge_fraction": float(
                            sum(int(mask.sum()) for mask in routed_masks.values())
                            / max(samples * len(routed_masks), 1)
                        ),
                    })

            if (route_captured and actuation_min_q <= q <= actuation_max_q
                    and guidance_stage in {"predictor", "both"}):
                corrected, _, diagnostics = shoppers_projection(
                    preview, None, False, active_override=routed_masks)
                if diagnostics["unique_rows"]:
                    encoded_before = codec.encode(preview)
                    encoded_after = codec.encode(corrected)
                    delta = encoded_after - encoded_before
                    norms = delta.norm(dim=1, keepdim=True)
                    if trust_radius > 0:
                        delta = delta * torch.clamp(
                            trust_radius / norms.clamp_min(1e-12), max=1.0)
                    denoised = denoised + strength * delta
                    step["latent_movement"] = float(
                        (strength * delta).norm(dim=1).sum().cpu())
                step.update(diagnostics)
            previous_masks = current_masks
            previous_signals = current_signals

        derivative = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * derivative
        if index < steps - 1:
            denoised_next = net(x_next, t_next).float()
            q_next = float(q_values[index + 1])
            guide_corrector = (
                guided and strength > 0
                and guidance_stage in {"corrector", "both"}
                and ((route_mode == "fixed" and q_min <= q_next <= q_max)
                     or (route_mode in {"auto_knee", "auto_valley"}
                         and route_captured
                         and actuation_min_q <= q_next <= actuation_max_q))
            )
            if guide_corrector:
                corrector_preview = codec.decode(denoised_next)
                active_override = routed_masks if route_mode != "fixed" else None
                corrected_next, next_masks, corrector_diag = shoppers_projection(
                    corrector_preview, previous_masks, require_persistence,
                    active_override=active_override)
                if corrector_diag["unique_rows"]:
                    encoded_before = codec.encode(corrector_preview)
                    encoded_after = codec.encode(corrected_next)
                    delta_next = encoded_after - encoded_before
                    norms = delta_next.norm(dim=1, keepdim=True)
                    if trust_radius > 0:
                        delta_next = delta_next * torch.clamp(
                            trust_radius / norms.clamp_min(1e-12), max=1.0)
                    denoised_next = denoised_next + strength * delta_next
                    movement = float((strength * delta_next).norm(dim=1).sum().cpu())
                    step["corrector_latent_movement"] = movement
                    step["latent_movement"] += movement
                step["corrector_unique_rows"] = corrector_diag["unique_rows"]
                if route_mode == "fixed":
                    previous_masks = next_masks
            derivative_next = (x_next - denoised_next) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * derivative + 0.5 * derivative_next)
        trace.append(step)

    frame = codec.decode(x_next)
    return frame, {
        "guided": guided,
        "strength": strength,
        "q_min": q_min,
        "q_max": q_max,
        "trust_radius": trust_radius,
        "require_persistence": require_persistence,
        "route_mode": route_mode,
        "actuation_min_q": actuation_min_q,
        "actuation_max_q": actuation_max_q,
        "guidance_stage": guidance_stage,
        "stability_threshold": stability_threshold,
        "auto_route_q": next((step["auto_route_q"] for step in trace
                              if "auto_route_q" in step), None),
        "total_unique_row_events": int(sum(step["unique_rows"] for step in trace)),
        "total_latent_movement": float(sum(step["latent_movement"] for step in trace)),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabsyn-repo", type=Path, required=True)
    parser.add_argument("--dataset", default="shoppers", choices=["shoppers"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--strength", type=float, default=.1)
    parser.add_argument("--q-min", type=float, default=.20)
    parser.add_argument("--q-max", type=float, default=.55)
    parser.add_argument("--trust-radius", type=float, default=.5)
    parser.add_argument("--require-persistence", action="store_true")
    parser.add_argument("--route-mode", choices=["fixed", "auto_knee", "auto_valley"], default="fixed")
    parser.add_argument("--actuation-min-q", type=float, default=0.0)
    parser.add_argument("--actuation-max-q", type=float, default=.15)
    parser.add_argument("--stability-threshold", type=float, default=.005)
    parser.add_argument("--guidance-stage", choices=["predictor", "corrector", "both"], default="predictor")
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional diffusion state_dict; defaults to the official dataset model.pt.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    codec = TabSynCodec(args.tabsyn_repo.resolve(), args.dataset, device)

    from tabsyn.model import MLPDiffusion, Model
    denoiser = MLPDiffusion(codec.train_z.shape[1], 1024).to(device)
    model = Model(denoise_fn=denoiser, hid_dim=codec.train_z.shape[1]).to(device)
    checkpoint = args.checkpoint or (Path(codec.diffusion_ckpt) / "model.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    net = model.denoise_fn_D.eval()

    baseline, baseline_trace = sample_once(
        net, codec, args.samples, args.steps, args.seed, 0.0,
        args.q_min, args.q_max, args.trust_radius,
        args.require_persistence, args.route_mode, args.actuation_min_q,
        args.actuation_max_q, args.stability_threshold, args.guidance_stage,
        guided=False)
    guided, guided_trace = sample_once(
        net, codec, args.samples, args.steps, args.seed, args.strength,
        args.q_min, args.q_max, args.trust_radius,
        args.require_persistence, args.route_mode, args.actuation_min_q,
        args.actuation_max_q, args.stability_threshold, args.guidance_stage,
        guided=True)
    baseline.to_csv(args.output / "tabsyn_baseline.csv", index=False)
    guided.to_csv(args.output / "tabsyn_dynamic_graph.csv", index=False)
    (args.output / "trace.json").write_text(json.dumps({
        "baseline": baseline_trace, "guided": guided_trace}, indent=2))
    (args.output / "run_metadata.json").write_text(json.dumps({
        "dataset": args.dataset, "samples": args.samples, "steps": args.steps,
        "seed": args.seed, "checkpoint": str(checkpoint),
        "method": "TabSyn clean-latent decode/project/re-encode add-on",
        "route_mode": args.route_mode, "actuation_min_q": args.actuation_min_q,
        "actuation_max_q": args.actuation_max_q,
        "guidance_stage": args.guidance_stage,
    }, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "row_events": guided_trace["total_unique_row_events"],
        "latent_movement": guided_trace["total_latent_movement"],
    }, indent=2))


if __name__ == "__main__":
    main()
