#!/usr/bin/env python3
"""Harpoon-style categorical energy guidance for a frozen TabDiff graph.

This is deliberately labelled a proxy rather than an official reproduction.
Harpoon's public method applies a differentiable constraint loss to predicted
clean states throughout reverse diffusion.  TabDiff's categorical process is
absorbing and has no continuous categorical state on which to take the same
gradient, so this implementation applies the corresponding target-logit
energy before each discrete reverse update.  The graph catalog is fixed; its
antecedents and violations are re-evaluated from x-hat-0 at every step.

Two policies are exposed. ``hard`` guides every currently violated row-edge;
``calibrated`` guides only violation mass above the train-only Wilson target.
The latter is a stronger soft-constraint baseline for empirical graphs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

from conditioned_tabdiff_adapter import decode, seed_all
from subgroup_projection_online import PreviewModel
from tabdiff_probe import preprocess_dataset
from typed_dynamic_graph_guidance import edge_masks, graph_metrics


def categorical_layout(train_data, info: dict) -> tuple[list[str], dict[str, int], list[int]]:
    names = ([info["column_names"][index] for index in info["target_col_idx"]]
             + [info["column_names"][index] for index in info["cat_col_idx"]])
    sizes = [int(value) for value in train_data.categories]
    return names, {name: index for index, name in enumerate(names)}, sizes


def categorical_code(train_data, position: int, value: object) -> int:
    categories = train_data.cat_transform.named_steps["ordinal"].categories_[position]
    matches = np.flatnonzero(np.asarray(categories).astype(str) == str(value))
    if len(matches) != 1:
        raise ValueError(f"categorical target {value!r} is not unique at internal position {position}")
    return int(matches[0])


def target_spec(edge: dict, train_data, positions: dict[str, int], sizes: list[int]) -> tuple[int, int, int]:
    requirements = edge.get("requirements", [])
    if len(requirements) != 1:
        raise ValueError(f"proxy expects one requirement per edge: {edge['candidate_id']}")
    atom = requirements[0]
    column = atom["left_column"]
    if atom.get("right_kind") != "literal" or atom.get("operator") != "==" or column not in positions:
        raise ValueError(f"proxy currently supports categorical literal equality only: {edge['candidate_id']}")
    position = positions[column]
    code = categorical_code(train_data, position, atom["right_value"])
    offset = int(sum(size + 1 for size in sizes[:position]))
    return position, code, offset


def wilson_upper(edge: dict, z: float) -> float:
    calibration = edge.get("calibration", {})
    support = int(calibration.get("support", 0))
    violations = int(calibration.get("violations", 0))
    if support <= 0:
        return float(calibration.get("violation_rate", 0.0))
    p = violations / support
    denominator = 1.0 + z * z / support
    center = p + z * z / (2.0 * support)
    radius = z * math.sqrt(p * (1.0 - p) / support + z * z / (4.0 * support * support))
    return float((center + radius) / denominator)


def distribution_distance(sample: pd.DataFrame, reference: pd.DataFrame, info: dict) -> tuple[float, float]:
    numeric = [info["column_names"][index] for index in info["num_col_idx"]]
    categorical = [info["column_names"][index] for index in info["cat_col_idx"]]
    ks = [ks_2samp(pd.to_numeric(sample[column], errors="coerce"),
                   pd.to_numeric(reference[column], errors="coerce")).statistic
          for column in numeric]
    tv = []
    for column in categorical:
        left = sample[column].astype(str).value_counts(normalize=True)
        right = reference[column].astype(str).value_counts(normalize=True)
        values = left.index.union(right.index)
        tv.append(0.5 * float((left.reindex(values, fill_value=0) -
                               right.reindex(values, fill_value=0)).abs().sum()))
    return float(np.mean(ks)), float(np.mean(tv))


@torch.no_grad()
def sample_guided(diffusion, preview: PreviewModel, train_data, info: dict,
                  edges: list[dict], samples: int, strength: float,
                  policy: str, active_min_t: float, active_max_t: float,
                  wilson_z: float) -> tuple[torch.Tensor, dict]:
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num = diffusion.num_schedule.total_noise(times)
    sigma_cat = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num); sigma_num_next[1:] = sigma_num[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat); sigma_cat_next[1:] = sigma_cat[:-1]
    if diffusion.sampler_params["stochastic_sampler"] or diffusion.sampler_params["second_order_correction"]:
        raise ValueError("proxy expects deterministic first-order sampling")

    names, positions, sizes = categorical_layout(train_data, info)
    del names
    specs = {edge["candidate_id"]: target_spec(edge, train_data, positions, sizes) for edge in edges}
    targets = {edge["candidate_id"]: wilson_upper(edge, wilson_z) for edge in edges}
    numeric = torch.randn((samples, diffusion.num_numerical_features), device=device) * sigma_num[-1]
    categorical = diffusion._sample_masked_prior(samples, len(diffusion.num_classes))
    trace = []
    total_guided = total_editable = 0

    for index in reversed(range(diffusion.num_timesteps)):
        t = float(times[index])
        # Match ``UnifiedCtimeDiffusion.edm_update`` exactly, including random
        # draws multiplied by zero in deterministic mode. Those draws still
        # advance the RNG and determine later categorical samples.
        numeric_hat = numeric + (sigma_num[index].square() - sigma_num[index].square()).sqrt() * torch.randn_like(numeric)
        move_chance = -torch.expm1(sigma_cat[index] - sigma_cat[index])
        categorical_hat, _ = diffusion.q_xt(categorical, move_chance)
        categorical_oh = diffusion.to_one_hot(categorical_hat).to(numeric.dtype)
        denoised, raw_logits = preview.base(
            numeric_hat.float(), categorical_oh, times[index].squeeze().repeat(samples),
            sigma=sigma_num[index].unsqueeze(0).repeat(samples, 1),
        )
        direction = (numeric_hat - denoised) / sigma_num[index]
        numeric_next = numeric_hat + (sigma_num_next[index] - sigma_num[index]) * direction
        codes = preview.predicted_codes(categorical_oh, raw_logits)
        frame = decode(train_data, torch.cat([denoised, codes.float()], dim=1).cpu(), info)

        guided_logits = raw_logits.detach().clone()
        step_guided = step_editable = step_active = step_bad = 0
        if active_min_t <= t <= active_max_t and strength > 0:
            for edge in edges:
                edge_id = edge["candidate_id"]
                active, _, bad = edge_masks(frame, edge)
                active_indices = np.flatnonzero(active)
                bad_indices = np.flatnonzero(bad)
                step_active += len(active_indices); step_bad += len(bad_indices)
                selected = bad_indices
                position, target_code, offset = specs[edge_id]
                if policy == "calibrated" and len(bad_indices):
                    allowed = int(math.floor(targets[edge_id] * len(active_indices)))
                    excess = max(0, len(bad_indices) - allowed)
                    if excess == 0:
                        selected = np.empty(0, dtype=int)
                    elif excess < len(bad_indices):
                        target_score = raw_logits[bad_indices, offset + target_code].detach().cpu().numpy()
                        selected = bad_indices[np.argsort(target_score)[:excess]]
                if len(selected):
                    selected_tensor = torch.as_tensor(selected, device=device, dtype=torch.long)
                    # Absorbing discrete diffusion can only respond through a
                    # logit while this feature is still masked. Report both the
                    # requested and actually editable row-edge counts.
                    editable = categorical_hat[selected_tensor, position].eq(int(sizes[position]))
                    editable_rows = selected_tensor[editable]
                    if len(editable_rows):
                        guided_logits[editable_rows, offset + target_code] += float(strength)
                    step_guided += len(selected); step_editable += len(editable_rows)

        logits = diffusion._subs_parameterization(guided_logits, categorical_hat)
        alpha_t = torch.exp(-sigma_cat[index]).unsqueeze(0).repeat(samples, 1)
        alpha_s = torch.exp(-sigma_cat_next[index]).unsqueeze(0).repeat(samples, 1)
        categorical_next, _ = diffusion._mdlm_update(logits, categorical_hat, alpha_t, alpha_s)
        numeric, categorical = numeric_next.detach(), categorical_next.detach()
        total_guided += step_guided; total_editable += step_editable
        trace.append({"reverse_step": diffusion.num_timesteps - 1 - index, "t": t,
                      "active_row_edges": step_active, "violated_row_edges": step_bad,
                      "requested_guidance": step_guided, "editable_guidance": step_editable})

    if not torch.all(categorical < diffusion.mask_index):
        raise RuntimeError("categorical mask remained at final step")
    return torch.cat([numeric, categorical.float()], dim=1).cpu(), {
        "policy": policy, "strength": float(strength), "active_window": [active_min_t, active_max_t],
        "requested_guidance": int(total_guided), "editable_guidance": int(total_editable),
        "wilson_z": float(wilson_z), "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261376)
    parser.add_argument("--strengths", default="0,1,2,4,8")
    parser.add_argument("--policies", default="hard,calibrated")
    parser.add_argument("--active-min-t", type=float, default=0.0)
    parser.add_argument("--active-max-t", type=float, default=1.0)
    parser.add_argument("--wilson-z", type=float, default=1.0)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    started = time.perf_counter()
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, args.dataset)
    heldout = decode(train_data, test_data.X, info)
    edges = json.loads(args.graph.read_text())
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["denoise_fn"], strict=True)
    preview = PreviewModel(model, train_data.categories)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column",
                            cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, preview, None,
        **diffusion_params, device=torch.device("cuda"),
    ).cuda()
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()

    sample_dir = args.output / "samples"; sample_dir.mkdir()
    common_seed = args.seed + 100000
    seed_all(common_seed)
    unconditional = decode(train_data, diffusion.sample(args.samples), info)
    unconditional.to_csv(sample_dir / "unconditional.csv", index=False)
    base_graph = graph_metrics(unconditional, heldout, edges)
    base_ks, base_tv = distribution_distance(unconditional, heldout, info)
    rows = [{"method": "unconditional", **base_graph, "numeric_ks": base_ks,
             "categorical_tv": base_tv, "requested_guidance": 0, "editable_guidance": 0,
             "runtime_seconds": 0.0}]

    strengths = [float(value) for value in args.strengths.split(",")]
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    for policy in policies:
        if policy not in {"hard", "calibrated"}:
            raise ValueError(policy)
        for strength in strengths:
            seed_all(common_seed); tick = time.perf_counter()
            encoded, trace = sample_guided(
                diffusion, preview, train_data, info, edges, args.samples, strength, policy,
                args.active_min_t, args.active_max_t, args.wilson_z,
            )
            frame = decode(train_data, encoded, info)
            label = f"harpoon_{policy}_w{strength:g}"
            frame.to_csv(sample_dir / f"{label}.csv", index=False)
            metrics = graph_metrics(frame, heldout, edges)
            ks, tv = distribution_distance(frame, heldout, info)
            elapsed = time.perf_counter() - tick
            rows.append({"method": label, **metrics, "numeric_ks": ks, "categorical_tv": tv,
                         "requested_guidance": trace["requested_guidance"],
                         "editable_guidance": trace["editable_guidance"],
                         "runtime_seconds": elapsed})
            trace["runtime_seconds"] = elapsed
            (args.output / f"trace_{label}.json").write_text(json.dumps(trace, indent=2))

    metrics = pd.DataFrame(rows); metrics.to_csv(args.output / "method_metrics.csv", index=False)
    metadata = {
        "label": "Harpoon-style graph-energy proxy; not an official Harpoon reproduction",
        "dataset": args.dataset, "graph": str(args.graph), "edges": len(edges),
        "samples": args.samples, "steps": args.steps, "seed": args.seed,
        "strengths": strengths, "policies": policies,
        "active_window": [args.active_min_t, args.active_max_t], "wilson_z": args.wilson_z,
        "categorical_response": "target-logit energy before absorbing discrete update",
        "runtime_seconds": time.perf_counter() - started, "api_calls": 0,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    lines = ["# Harpoon-style graph proxy", "",
             "This is a local rerun variant, not an official Harpoon reproduction.", "",
             "| Method | Macro violation | Held-out error | Calibration error | Numeric KS | TV | Requested | Editable | Runtime |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.method} | {row.macro_violation:.4f} | {row.heldout_violation_error:.4f} | "
                     f"{row.calibration_violation_error:.4f} | {row.numeric_ks:.4f} | "
                     f"{row.categorical_tv:.4f} | {row.requested_guidance} | "
                     f"{row.editable_guidance} | {row.runtime_seconds:.2f} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
