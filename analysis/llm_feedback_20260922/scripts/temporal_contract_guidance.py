#!/usr/bin/env python3
"""Rare-contract sampling with time-varying classifier-free guidance."""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import normalized_mutual_info_score

from conditioned_tabdiff_adapter import ConditionedUniModMLP, ContractConditionedModel, decode, seed_all
from rejection_frontier import add_result, certify, contract_vector, contracts_for, satisfies
from tabdiff_probe import preprocess_dataset


SENSITIVE_COLUMN_RE = re.compile(r"age|sex|gender|race|ethnic|marital|relationship", re.IGNORECASE)


def dependency_neighbors(
    frame: pd.DataFrame, targets: set[str], k: int, excluded: set[str] | None = None
) -> tuple[set[str], list[tuple[str, float]]]:
    """Select a train-only dependency neighborhood using discretized NMI."""
    if k <= 0:
        return set(), []

    def discretize(series: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.98 and numeric.nunique(dropna=True) > 12:
            try:
                values = pd.qcut(numeric, q=min(10, numeric.nunique()), duplicates="drop").astype("string")
            except ValueError:
                values = numeric.round(6).astype("string")
        else:
            values = series.astype("string")
        return values.fillna("<MISSING>").to_numpy(str)

    excluded = excluded or set()
    target_values = {column: discretize(frame[column]) for column in targets}
    scores = []
    for column in frame.columns:
        if column in targets or column in excluded:
            continue
        candidate = discretize(frame[column])
        score = max(normalized_mutual_info_score(values, candidate) for values in target_values.values())
        scores.append((column, float(score)))
    scores.sort(key=lambda item: (-item[1], item[0]))
    selected = scores[:k]
    return {column for column, _ in selected}, selected


class TemporalContractModel(ContractConditionedModel):
    def __init__(
        self, *args, profile: str, peak: float, floor: float, cutoff: float,
        scope: str, scope_cutoff: float, numeric_positions: dict[str, int],
        categorical_slices: dict[str, slice], **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.profile = profile
        self.peak = float(peak)
        self.floor = float(floor)
        self.cutoff = float(cutoff)
        self.scope = scope
        self.scope_cutoff = float(scope_cutoff)
        self.numeric_positions = numeric_positions
        self.categorical_slices = categorical_slices
        self.target_columns: set[str] = set()
        self.graph_candidates: list[tuple[str, float]] = []
        self.dynamic_k = 0
        self.reset_graph_stats()

    def set_target_columns(self, columns: set[str]) -> None:
        unknown = columns - set(self.numeric_positions) - set(self.categorical_slices)
        if unknown:
            raise KeyError(f"unknown target columns: {sorted(unknown)}")
        self.target_columns = set(columns)

    def set_dynamic_graph(self, candidates: list[tuple[str, float]], k: int) -> None:
        """Set a train-derived graph prior whose active edges vary by row and timestep."""
        self.graph_candidates = list(candidates)
        self.dynamic_k = min(int(k), len(self.graph_candidates))
        self.reset_graph_stats()

    def reset_graph_stats(self) -> None:
        self.graph_steps = 0
        self.graph_rows = 0
        self.graph_edge_counts: dict[str, float] = {}
        self.graph_phase_counts: dict[str, dict[str, float]] = {
            "early": {}, "middle": {}, "late": {},
        }
        self.graph_phase_rows = {"early": 0, "middle": 0, "late": 0}
        self.graph_change_sum = 0.0
        self.graph_change_pairs = 0
        self._last_graph_selection: torch.Tensor | None = None

    def graph_summary(self) -> dict:
        denominator = max(self.graph_rows, 1)
        phase_rates = {}
        for phase, counts in self.graph_phase_counts.items():
            phase_denominator = max(self.graph_phase_rows[phase], 1)
            phase_rates[phase] = {
                name: count / phase_denominator for name, count in counts.items()
            }
        return {
            "steps": self.graph_steps,
            "rows": self.graph_rows,
            "edge_selection_rate": {
                name: count / denominator for name, count in self.graph_edge_counts.items()
            },
            "phase_edge_selection_rate": phase_rates,
            "mean_edge_change_rate": (
                self.graph_change_sum / self.graph_change_pairs if self.graph_change_pairs else 0.0
            ),
        }

    def _dynamic_graph_masks(
        self, active, numeric_delta: torch.Tensor, categorical_delta: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select a per-row active subgraph from conditional x-hat response and uncertainty."""
        batch = numeric_delta.shape[0]
        numeric_mask = torch.zeros_like(numeric_delta)
        categorical_mask = torch.zeros_like(categorical_delta)
        for column in self.target_columns:
            if column in self.numeric_positions:
                numeric_mask[:, self.numeric_positions[column]] = 1
            else:
                categorical_mask[:, self.categorical_slices[column]] = 1
        if self.dynamic_k <= 0:
            return numeric_mask, categorical_mask

        scores = []
        for column, graph_weight in self.graph_candidates:
            if column in self.numeric_positions:
                response = numeric_delta[:, self.numeric_positions[column]].abs()
                reliability = torch.ones_like(response)
            else:
                column_slice = self.categorical_slices[column]
                delta = categorical_delta[:, column_slice]
                response = delta.square().mean(dim=1).sqrt()
                logits = active[1][:, column_slice]
                probabilities = logits.softmax(dim=1).clamp_min(1e-12)
                entropy = -(probabilities * probabilities.log()).sum(dim=1)
                entropy = entropy / max(math.log(logits.shape[1]), 1e-12)
                reliability = 0.25 + 0.75 * entropy
            scores.append(response * reliability * (0.25 + 0.75 * float(graph_weight)))
        score_matrix = torch.stack(scores, dim=1)
        selected_index = score_matrix.topk(self.dynamic_k, dim=1).indices
        selection = torch.zeros_like(score_matrix, dtype=torch.bool)
        selection.scatter_(1, selected_index, True)
        for index, (column, _) in enumerate(self.graph_candidates):
            selected = selection[:, index]
            if column in self.numeric_positions:
                numeric_mask[:, self.numeric_positions[column]] = selected.to(numeric_mask.dtype)
            else:
                categorical_mask[:, self.categorical_slices[column]] = selected[:, None].to(categorical_mask.dtype)

        detached = selection.detach()
        self.graph_steps += 1
        self.graph_rows += batch
        mean_t = float(t.detach().float().mean().item())
        phase = "early" if mean_t > 2 / 3 else ("middle" if mean_t > 1 / 3 else "late")
        self.graph_phase_rows[phase] += batch
        for index, (column, _) in enumerate(self.graph_candidates):
            count = float(detached[:, index].sum().item())
            self.graph_edge_counts[column] = self.graph_edge_counts.get(column, 0.0) + count
            phase_counts = self.graph_phase_counts[phase]
            phase_counts[column] = phase_counts.get(column, 0.0) + count
        if self._last_graph_selection is not None and self._last_graph_selection.shape == detached.shape:
            self.graph_change_sum += float((detached != self._last_graph_selection).float().mean().item())
            self.graph_change_pairs += 1
        self._last_graph_selection = detached.clone()
        return numeric_mask, categorical_mask

    def scale_at(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().clamp(0, 1)
        if self.profile == "constant":
            value = torch.full_like(t, self.peak)
        elif self.profile == "step_release":
            value = torch.where(t >= self.cutoff, self.peak, self.floor)
        elif self.profile == "linear_release":
            value = self.floor + (self.peak - self.floor) * t
        elif self.profile == "linear_ramp":
            value = self.floor + (self.peak - self.floor) * (1 - t)
        elif self.profile == "midpulse":
            value = self.floor + (self.peak - self.floor) * torch.sin(math.pi * t)
        else:
            raise ValueError(self.profile)
        return value.reshape(-1, 1)

    def forward(self, x_num, x_cat, t, sigma=None):
        condition = self.condition
        if condition is None:
            condition = torch.zeros(len(x_num), self.condition_dim, device=x_num.device, dtype=x_num.dtype)
        self.backbone.set_condition(condition)
        active = self.base_model(x_num, x_cat, t, sigma=sigma)
        if self.scope == "all" and self.profile == "constant" and math.isclose(self.peak, 1.0):
            return active
        self.backbone.set_condition(torch.zeros_like(condition))
        base = self.base_model(x_num, x_cat, t, sigma=sigma)
        self.backbone.set_condition(condition)
        scale = self.scale_at(t).to(device=x_num.device, dtype=x_num.dtype)
        numeric_delta = active[0] - base[0]
        categorical_delta = active[1] - base[1]
        if self.scope in {"target", "target_neighbors", "all_then_target", "dynamic_graph"}:
            if self.scope == "dynamic_graph":
                numeric_mask, categorical_mask = self._dynamic_graph_masks(
                    active, numeric_delta, categorical_delta, t
                )
            else:
                numeric_mask = torch.zeros((1, numeric_delta.shape[1]), device=x_num.device, dtype=x_num.dtype)
                categorical_mask = torch.zeros(
                    (1, categorical_delta.shape[1]), device=x_num.device, dtype=x_num.dtype
                )
                for column in self.target_columns:
                    if column in self.numeric_positions:
                        numeric_mask[:, self.numeric_positions[column]] = 1
                    else:
                        categorical_mask[:, self.categorical_slices[column]] = 1
            target_numeric_delta = numeric_delta * numeric_mask
            target_categorical_delta = categorical_delta * categorical_mask
            if self.scope in {"target", "target_neighbors", "dynamic_graph"}:
                numeric_delta = target_numeric_delta
                categorical_delta = target_categorical_delta
            else:
                # Sampling runs from noisy t=1 to clean t=0.  First steer the
                # whole joint record, then restrict late refinement to the
                # explicitly requested columns.
                late = (t.float() <= self.scope_cutoff).reshape(-1, 1)
                numeric_delta = torch.where(late, target_numeric_delta, numeric_delta)
                categorical_delta = torch.where(late, target_categorical_delta, categorical_delta)
        return base[0] + scale * numeric_delta, base[1] + scale * categorical_delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", choices=["adult", "default", "shoppers"], required=True)
    parser.add_argument(
        "--catalog", choices=["numeric", "mixed", "support_numeric", "support_mixed"], required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["constant", "step_release", "linear_release", "linear_ramp", "midpulse"], required=True)
    parser.add_argument("--peak", type=float, required=True)
    parser.add_argument("--floor", type=float, required=True)
    parser.add_argument("--cutoff", type=float, default=0.5)
    parser.add_argument(
        "--scope", choices=["all", "target", "target_neighbors", "all_then_target", "dynamic_graph"], default="all"
    )
    parser.add_argument("--scope-cutoff", type=float, default=0.25)
    parser.add_argument("--neighbor-k", type=int, default=2)
    parser.add_argument("--graph-candidate-k", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    started = time.perf_counter()
    seed_all(args.seed)
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    device = torch.device("cuda")
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, args.dataset)
    train = decode(train_data, train_data.X, info)
    heldout = decode(train_data, test_data.X, info)
    numeric = [info["column_names"][index] for index in info["num_col_idx"]]
    contracts = contracts_for(args.dataset, args.catalog, train)
    saved = torch.load(args.adapter, map_location="cpu", weights_only=False)
    vectors = {contract["contract_id"]: contract_vector(contract, args.dataset, args.catalog) for contract in contracts}
    condition_dim = int(saved["condition_dim"])
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    backbone = ConditionedUniModMLP(condition_dim=condition_dim, **params)
    base = Model(backbone, **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = base.load_state_dict(checkpoint["denoise_fn"], strict=False)
    unexpected_missing = [key for key in incompatible.missing_keys if "condition_projection" not in key]
    if incompatible.unexpected_keys or unexpected_missing:
        raise RuntimeError(str(incompatible))
    backbone.mlp.condition_projection.load_state_dict(saved["condition_projection"])
    backbone.mlp.mlp.load_state_dict(saved["latent_mlp"])
    numeric_names = [info["column_names"][index] for index in info["num_col_idx"]]
    category_names = [info["column_names"][index] for index in info["target_col_idx"] + info["cat_col_idx"]]
    evaluation_targets = {info["column_names"][index] for index in info["target_col_idx"]}
    protected_columns = {name for name in train.columns if SENSITIVE_COLUMN_RE.search(str(name))}
    neighbor_exclusions = evaluation_targets | protected_columns
    numeric_positions = {name: index for index, name in enumerate(numeric_names)}
    categorical_slices = {}
    start = 0
    for name, size in zip(category_names, train_data.categories):
        categorical_slices[name] = slice(start, start + int(size) + 1)
        start += int(size) + 1
    model = TemporalContractModel(
        base, backbone, condition_dim, profile=args.profile, peak=args.peak, floor=args.floor,
        cutoff=args.cutoff, scope=args.scope, scope_cutoff=args.scope_cutoff,
        numeric_positions=numeric_positions,
        categorical_slices=categorical_slices,
    ).to(device)
    diffusion_parameters = copy.deepcopy(config["diffusion_params"])
    diffusion_parameters.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_parameters["sampler_params"] = copy.deepcopy(diffusion_parameters["sampler_params"])
    diffusion_parameters["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, model, None, **diffusion_parameters, device=device
    ).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    diffusion.eval()

    rows = []
    budget_rows = []
    calls_per_trajectory = 1 if args.scope == "all" and args.profile == "constant" and math.isclose(args.peak, 1.0) else 2
    for ordinal, contract in enumerate(contracts):
        reference = heldout[satisfies(heldout, contract)].copy()
        condition = torch.tensor(vectors[contract["contract_id"]], dtype=torch.float32, device=device).repeat(args.samples, 1)
        contract_columns = {column for column, _, _ in contract["predicates"]}
        selected_neighbors: list[tuple[str, float]] = []
        active_columns = set(contract_columns)
        if args.scope == "target_neighbors":
            neighbors, selected_neighbors = dependency_neighbors(
                train, contract_columns, args.neighbor_k, excluded=neighbor_exclusions - contract_columns
            )
            active_columns.update(neighbors)
        elif args.scope == "dynamic_graph":
            _, raw_candidates = dependency_neighbors(
                train, contract_columns, args.graph_candidate_k,
                excluded=neighbor_exclusions - contract_columns,
            )
            if raw_candidates:
                maximum = max(score for _, score in raw_candidates)
                selected_neighbors = [
                    (column, score / maximum if maximum > 0 else 1.0)
                    for column, score in raw_candidates
                ]
            active_columns.update(column for column, _ in selected_neighbors)
        model.set_target_columns(contract_columns if args.scope == "dynamic_graph" else active_columns)
        model.set_dynamic_graph(selected_neighbors if args.scope == "dynamic_graph" else [], args.neighbor_k)
        model.set_condition(condition, 1.0)
        seed_all(args.seed + 300_000 + ordinal)
        with torch.no_grad():
            generated = decode(train_data, diffusion.sample(args.samples), info)
        valid = satisfies(generated, contract)
        method = f"{args.profile}_p{args.peak:g}_f{args.floor:g}_c{args.cutoff:g}"
        if args.scope == "target":
            method += "_target"
        elif args.scope == "target_neighbors":
            method += f"_neighbors{args.neighbor_k}"
        elif args.scope == "all_then_target":
            method += f"_all2target_sc{args.scope_cutoff:g}"
        elif args.scope == "dynamic_graph":
            method += f"_dyngraph{args.neighbor_k}of{args.graph_candidate_k}"
        add_result(rows, generated, reference, numeric, contract, method, args.dataset, args.catalog, args.seed, args.samples)
        add_result(rows, generated[valid].copy(), reference, numeric, contract, method + "_accepted", args.dataset, args.catalog, args.seed, args.samples)
        certified, edit_rate, numeric_distortion = certify(generated, contract, heldout)
        add_result(
            rows, certified, reference, numeric, contract, method + "_certified", args.dataset, args.catalog,
            args.seed, args.samples, edit_rate, numeric_distortion,
        )
        probability = float(valid.mean())
        budget_rows.append({
            "dataset": args.dataset, "catalog": args.catalog, "split": contract["split"], "seed": args.seed,
            "contract_id": contract["contract_id"], "profile": args.profile, "peak": args.peak,
            "floor": args.floor, "cutoff": args.cutoff, "scope_cutoff": args.scope_cutoff,
            "method": method,
            "actuation_scope": args.scope,
            "neighbor_k": args.neighbor_k,
            "graph_candidate_k": args.graph_candidate_k,
            "active_columns": json.dumps(sorted(active_columns)),
            "selected_neighbors": json.dumps(selected_neighbors),
            "dynamic_graph_summary": json.dumps(model.graph_summary(), sort_keys=True),
            "neighbor_exclusions": json.dumps(sorted(neighbor_exclusions - contract_columns)),
            "single_trajectory_success": probability,
            "expected_trajectories_per_valid": math.inf if probability == 0 else 1.0 / probability,
            "denoiser_calls_per_trajectory": calls_per_trajectory,
            "expected_denoiser_calls_per_valid": math.inf if probability == 0 else calls_per_trajectory / probability,
            "heldout_support": float(len(reference) / len(heldout)),
        })
    metrics = pd.DataFrame(rows)
    metrics["profile"] = args.profile
    metrics["peak"] = args.peak
    metrics["floor"] = args.floor
    metrics["cutoff"] = args.cutoff
    metrics["denoiser_calls_per_trajectory"] = calls_per_trajectory
    metrics["actuation_scope"] = args.scope
    metrics["scope_cutoff"] = args.scope_cutoff
    metrics["neighbor_k"] = args.neighbor_k
    metrics["graph_candidate_k"] = args.graph_candidate_k
    metrics.to_csv(args.output / "temporal_metrics.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(args.output / "temporal_budget.csv", index=False)
    summary = metrics.groupby("method").agg(
        contracts=("contract_id", "nunique"), satisfaction=("satisfaction", "mean"),
        numeric_ks=("numeric_ks", "mean"), categorical_tv=("categorical_tv", "mean"),
        edit_rate=("edit_rate", "mean"), numeric_distortion=("numeric_distortion", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = {
        "status": "executed", "dataset": args.dataset, "catalog": args.catalog, "seed": args.seed,
        "profile": args.profile, "peak": args.peak, "floor": args.floor, "cutoff": args.cutoff,
        "actuation_scope": args.scope,
        "scope_cutoff": args.scope_cutoff,
        "neighbor_k": args.neighbor_k,
        "graph_candidate_k": args.graph_candidate_k,
        "samples_per_contract": args.samples, "steps": args.steps, "calls_per_trajectory": calls_per_trajectory,
        "runtime_seconds": time.perf_counter() - started,
        "scope": "paired-seed temporal contract guidance; no retraining",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
