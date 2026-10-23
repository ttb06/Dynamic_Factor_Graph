#!/usr/bin/env python3
"""Dynamic graph-triggered conditional repainting for pretrained TabDiff.

A typed graph is evaluated on a predicted-clean preview at a chosen reverse
time. Only consequent columns of violated, sufficiently hard edges are
reopened; TabDiff conditionally regenerates those columns while holding the
rest of the preview fixed. Several stochastic proposals compete with the
untouched base trajectory under a frozen graph/density/edit score.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from conditioned_tabdiff_adapter import decode
from graph_conditioned_residual_denoiser import calibrated_minimal_final_repair, load_backbone, seed_all
from subgroup_projection_online import PreviewModel
from typed_dynamic_graph_guidance import TypedActuator, edge_masks, graph_metrics


def eligible_edges(edges: list[dict], max_calibration_rate: float) -> list[dict]:
    return [edge for edge in edges
            if edge.get("usage_role", "actuate") == "actuate"
            and len(edge.get("requirements", [])) == 1
            and float(edge.get("calibration", {}).get("violation_rate", 0.0)) <= max_calibration_rate]


@torch.no_grad()
def sample_base_and_preview(diffusion, samples: int, checkpoint: float, seed: int):
    """Sample once and retain the predicted-clean row nearest checkpoint."""
    seed_all(seed); device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sigma_num_cur = diffusion.num_schedule.total_noise(times)
    sigma_cat_cur = diffusion.cat_schedule.total_noise(times)
    sigma_num_next = torch.zeros_like(sigma_num_cur); sigma_num_next[1:] = sigma_num_cur[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat_cur); sigma_cat_next[1:] = sigma_cat_cur[:-1]
    if diffusion.sampler_params["stochastic_sampler"] or diffusion.sampler_params["second_order_correction"]:
        raise ValueError("paired repainting requires deterministic first-order base sampling")
    numeric = torch.randn((samples, diffusion.num_numerical_features), device=device) * sigma_num_cur[-1]
    categorical = diffusion._sample_masked_prior(samples, len(diffusion.num_classes))
    base_model = diffusion._denoise_fn
    preview_model = PreviewModel(base_model, diffusion.num_classes).to(device).eval()
    diffusion._denoise_fn = preview_model
    wanted = int(torch.argmin(torch.abs(times.squeeze(-1) - checkpoint)).item())
    preview_encoded = None; preview_time = None; started = time.perf_counter()
    try:
        for index in reversed(range(diffusion.num_timesteps)):
            numeric, categorical, _ = diffusion.edm_update(
                numeric, categorical, index, times[index], times[index - 1] if index > 0 else None,
                times[index], sigma_num_cur[index], sigma_num_next[index], sigma_num_cur[index],
                sigma_cat_cur[index], sigma_cat_next[index], sigma_cat_cur[index])
            if index == wanted:
                if preview_model.numeric is None or preview_model.codes is None:
                    raise RuntimeError("preview model did not expose a clean estimate")
                preview_encoded = torch.cat(
                    [preview_model.numeric.detach(), preview_model.codes.detach().to(preview_model.numeric.dtype)], 1).cpu()
                preview_time = float(times[index])
    finally:
        diffusion._denoise_fn = base_model
    if preview_encoded is None:
        raise RuntimeError("requested preview was not captured")
    return (torch.cat([numeric, categorical], 1).cpu(), preview_encoded, preview_time,
            time.perf_counter() - started)


def requirement_target(edge: dict, numeric_names: list[str], categorical_names: list[str]) -> str | None:
    requirement = edge["requirements"][0]; left = requirement["left_column"]
    all_names = numeric_names + categorical_names
    if left not in all_names:
        return None
    if requirement.get("right_kind") == "column":
        right = requirement["right_column"]
        protected = {atom["left_column"] for atom in edge.get("applicability", [])}
        if left in protected and right in all_names:
            return right
    return left


def row_target_groups(frame: pd.DataFrame, edges: list[dict], numeric_names: list[str],
                      categorical_names: list[str], max_targets: int):
    priorities = {"hard": 3, "near_hard": 2, "soft": 1}
    per_row: dict[int, list[tuple[int, float, str]]] = defaultdict(list)
    triggered_edges: Counter[str] = Counter()
    for edge in edges:
        _, _, bad = edge_masks(frame, edge)
        target = requirement_target(edge, numeric_names, categorical_names)
        if target is None:
            continue
        priority = priorities.get(edge.get("classification", "soft"), 1)
        confidence = float(edge.get("confidence", 0.0))
        for row in np.flatnonzero(bad):
            per_row[int(row)].append((priority, confidence, target))
            triggered_edges[str(edge["candidate_id"])] += 1
    grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row, candidates in per_row.items():
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2])); targets = []
        for _, _, target in candidates:
            if target not in targets:
                targets.append(target)
            if len(targets) >= max_targets:
                break
        if targets:
            grouped[tuple(sorted(targets))].append(row)
    return {key: np.asarray(value, dtype=int) for key, value in grouped.items()}, triggered_edges


class FrozenProposalScore:
    """Train-only proposal score; held-out rows never enter selection."""

    def __init__(self, train_data, train: pd.DataFrame, info: dict, edges: list[dict],
                 graph_weight: float, density_weight: float, rarity_weight: float,
                 edit_weight: float, density_bank: int):
        self.train_data = train_data; self.edges = edges
        self.graph_weight = float(graph_weight); self.density_weight = float(density_weight)
        self.rarity_weight = float(rarity_weight); self.edit_weight = float(edit_weight)
        self.numeric_names = [info["column_names"][i] for i in info["num_col_idx"]]
        self.categorical_names = [info["column_names"][i] for i in info["target_col_idx"] + info["cat_col_idx"]]
        self.scales = {name: max(float(pd.to_numeric(train[name], errors="coerce").std()), 1e-6)
                       for name in self.numeric_names}
        bank = train_data.X[:, :train_data.d_numerical].cpu().numpy()
        if density_bank > 0 and len(bank) > density_bank:
            bank = bank[np.random.default_rng(9173).choice(len(bank), density_bank, replace=False)]
        self.knn = NearestNeighbors(n_neighbors=min(5, len(bank))).fit(bank)
        reference = self.knn.kneighbors(bank[:min(2048, len(bank))], return_distance=True)[0][:, -1]
        self.density_scale = max(float(np.median(reference)), 1e-6)
        self.category_probability = {name: train[name].astype(str).value_counts(normalize=True).to_dict()
                                     for name in self.categorical_names}
        self.edge_weight = {"hard": 1.0, "near_hard": .75, "soft": .35}

    def graph_energy(self, frame: pd.DataFrame) -> np.ndarray:
        result = np.zeros(len(frame), dtype=float)
        for edge in self.edges:
            _, _, bad = edge_masks(frame, edge)
            result += self.edge_weight.get(edge.get("classification", "soft"), .35) * bad
        return result

    def density(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame[self.numeric_names].apply(pd.to_numeric, errors="coerce")
        encoded = self.train_data.num_transform.transform(raw.fillna(raw.median()).to_numpy(float))
        return self.knn.kneighbors(encoded, return_distance=True)[0][:, -1] / self.density_scale

    def rarity(self, frame: pd.DataFrame, targets: tuple[str, ...]) -> np.ndarray:
        selected = [name for name in targets if name in self.categorical_names]
        if not selected:
            return np.zeros(len(frame), dtype=float)
        terms = []
        for name in selected:
            probabilities = self.category_probability[name]
            terms.append(np.asarray([-math.log(max(probabilities.get(str(value), 1e-6), 1e-6))
                                     for value in frame[name]], dtype=float))
        return np.mean(terms, axis=0)

    def edit(self, base: pd.DataFrame, candidate: pd.DataFrame, targets: tuple[str, ...]) -> np.ndarray:
        terms = []
        for name in targets:
            if name in self.numeric_names:
                left = pd.to_numeric(base[name], errors="coerce").to_numpy(float)
                right = pd.to_numeric(candidate[name], errors="coerce").to_numpy(float)
                terms.append(np.abs(left - right) / self.scales[name])
            else:
                terms.append((base[name].astype(str).to_numpy() != candidate[name].astype(str).to_numpy()).astype(float))
        return np.mean(terms, axis=0) if terms else np.zeros(len(base), dtype=float)

    def __call__(self, base: pd.DataFrame, candidate: pd.DataFrame, targets: tuple[str, ...]) -> np.ndarray:
        return (self.graph_weight * self.graph_energy(candidate) + self.density_weight * self.density(candidate)
                + self.rarity_weight * self.rarity(candidate, targets)
                + self.edit_weight * self.edit(base, candidate, targets))


@torch.no_grad()
def conditional_repaint(diffusion, train_data, info: dict, base_frame: pd.DataFrame,
                        condition_encoded: torch.Tensor, trigger_frame: pd.DataFrame,
                        edges: list[dict], scorer: FrozenProposalScore, branches: int,
                        rounds: int, max_targets: int, seed: int, accept_margin: float):
    numeric_names = scorer.numeric_names; categorical_names = scorer.categorical_names
    groups, triggered_edges = row_target_groups(
        trigger_frame, edges, numeric_names, categorical_names, max_targets)
    result = base_frame.copy(); branch_choices: Counter[int] = Counter(); generated = 0
    group_rows = {}; started = time.perf_counter()
    for ordinal, (targets, indices) in enumerate(sorted(groups.items())):
        subset = condition_encoded[indices]
        num_positions = [numeric_names.index(name) for name in targets if name in numeric_names]
        cat_positions = [categorical_names.index(name) for name in targets if name in categorical_names]
        base_subset = base_frame.iloc[indices].reset_index(drop=True)
        proposals = [base_subset.copy()]
        for branch in range(branches):
            seed_all(seed + ordinal * 1009 + branch)
            sampled = diffusion.sample_impute(
                subset[:, :train_data.d_numerical].to(diffusion.device),
                subset[:, train_data.d_numerical:].long().to(diffusion.device),
                num_positions, cat_positions, rounds, "x_t", 0.0, 0.0)
            decoded = decode(train_data, sampled, info)
            candidate = base_subset.copy()
            for name in targets:
                candidate[name] = decoded[name].to_numpy()
            proposals.append(candidate)
        scores = np.stack([scorer(base_subset, proposal, targets) for proposal in proposals], 0)
        scores[~np.isfinite(scores)] = np.inf
        choice = np.argmin(scores, axis=0)
        improvement = scores[0] - scores[choice, np.arange(len(indices))]
        choice[improvement < accept_margin] = 0
        for branch, proposal in enumerate(proposals):
            selected = choice == branch
            if not selected.any():
                continue
            branch_choices[branch] += int(selected.sum())
            if branch:
                destination = indices[selected]
                for name in targets:
                    result.loc[destination, name] = proposal.loc[selected, name].to_numpy()
        generated += branches * len(indices)
        group_rows["|".join(targets)] = int(len(indices))
    diagnostics = {
        "triggered_rows": int(sum(len(value) for value in groups.values())),
        "target_groups": len(groups), "group_rows": group_rows,
        "triggered_edges": dict(triggered_edges), "proposal_rows_generated": generated,
        "branch_choices": {str(key): value for key, value in branch_choices.items()},
        "unique_rows_changed": int(sum(value for key, value in branch_choices.items() if key != 0)),
        "runtime_seconds": time.perf_counter() - started,
    }
    return result, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trigger", choices=["mid", "final"], default="mid")
    parser.add_argument("--proposal-source", choices=["trigger", "final"], default="trigger",
                        help="State conditioned on when drawing correction proposals.")
    parser.add_argument("--checkpoint-time", type=float, default=.25)
    parser.add_argument("--samples", type=int, default=4096); parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--branches", type=int, default=4); parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=3); parser.add_argument("--max-calibration-rate", type=float, default=.01)
    parser.add_argument("--graph-weight", type=float, default=20.0); parser.add_argument("--density-weight", type=float, default=.5)
    parser.add_argument("--rarity-weight", type=float, default=.1); parser.add_argument("--edit-weight", type=float, default=.5)
    parser.add_argument("--accept-margin", type=float, default=0.0); parser.add_argument("--density-bank", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261440)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True); sample_dir = args.output / "samples"; sample_dir.mkdir()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    device = torch.device("cuda"); started = time.perf_counter()
    diffusion, train_data, test_data, info, _ = load_backbone(
        args.repo, args.dataset, args.config, args.checkpoint, device, args.steps)
    edges = json.loads(args.graph.read_text())
    selected_edges = eligible_edges(edges, args.max_calibration_rate)
    if not selected_edges:
        raise RuntimeError("no graph edge passed the actuation/calibration filter")
    train = decode(train_data, train_data.X, info)
    heldout = decode(train_data, test_data.X, info)
    common_seed = args.seed + 100000
    base_encoded, preview_encoded, actual_time, base_runtime = sample_base_and_preview(
        diffusion, args.samples, args.checkpoint_time, common_seed)
    base_frame = decode(train_data, base_encoded, info)
    preview_frame = decode(train_data, preview_encoded, info)
    base_frame.to_csv(sample_dir / "tabdiff.csv", index=False)
    preview_frame.to_csv(sample_dir / "trigger_preview.csv", index=False)
    trigger_encoded = preview_encoded if args.trigger == "mid" else base_encoded
    trigger_frame = preview_frame if args.trigger == "mid" else base_frame
    condition_encoded = base_encoded if args.proposal_source == "final" else trigger_encoded
    scorer = FrozenProposalScore(
        train_data, train, info, selected_edges, args.graph_weight, args.density_weight,
        args.rarity_weight, args.edit_weight, args.density_bank)
    repaint, diagnostics = conditional_repaint(
        diffusion, train_data, info, base_frame, condition_encoded, trigger_frame,
        selected_edges, scorer, args.branches, args.rounds, args.max_targets,
        args.seed + 200000, args.accept_margin)
    method = f"graph_repaint_{args.trigger}"
    repaint.to_csv(sample_dir / f"{method}.csv", index=False)
    actuator = TypedActuator(edges, train_data, train, info)
    base_repair, base_repair_diag = calibrated_minimal_final_repair(base_frame, actuator)
    repaint_repair, repaint_repair_diag = calibrated_minimal_final_repair(repaint, actuator)
    base_repair.to_csv(sample_dir / "calibrated_final_repair.csv", index=False)
    repaint_repair.to_csv(sample_dir / f"{method}_plus_repair.csv", index=False)
    rows = [
        {"method": "tabdiff", **graph_metrics(base_frame, heldout, edges), "runtime_seconds": base_runtime},
        {"method": "calibrated_final_repair", **graph_metrics(base_repair, heldout, edges),
         "runtime_seconds": base_runtime, **base_repair_diag},
        {"method": method, **graph_metrics(repaint, heldout, edges),
         "runtime_seconds": base_runtime + diagnostics["runtime_seconds"], **diagnostics},
        {"method": f"{method}_plus_repair", **graph_metrics(repaint_repair, heldout, edges),
         "runtime_seconds": base_runtime + diagnostics["runtime_seconds"], **repaint_repair_diag},
    ]
    pd.DataFrame(rows).to_csv(args.output / "mechanism_metrics.csv", index=False)
    metadata = {
        "method": "dynamic_graph_conditional_repainting", "dataset": args.dataset,
        "trigger": args.trigger, "requested_checkpoint": args.checkpoint_time,
        "actual_checkpoint": actual_time, "graph_edges": len(edges),
        "eligible_edges": len(selected_edges), "api_calls": 0,
        "selection_uses_heldout": False, "diagnostics": diagnostics,
        "runtime_seconds": time.perf_counter() - started,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
