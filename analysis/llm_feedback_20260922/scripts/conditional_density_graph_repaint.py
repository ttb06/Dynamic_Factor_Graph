#!/usr/bin/env python3
"""Middle-state graph repainting with a context-conditional action selector.

The frozen baseline scores proposals using global train density.  This branch
adds a target likelihood conditioned on the row columns that remain fixed.
It therefore resolves ambiguous repairs by preferring corrections observed in
similar real training contexts, without using held-out rows or target labels.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from conditioned_tabdiff_adapter import decode
from dynamic_graph_conditional_repaint import (
    FrozenProposalScore,
    conditional_repaint,
    eligible_edges,
    sample_base_and_preview,
)
from graph_conditioned_residual_denoiser import (
    calibrated_minimal_final_repair,
    load_backbone,
)
from typed_dynamic_graph_guidance import TypedActuator, graph_metrics


class ConditionalProposalScore(FrozenProposalScore):
    """Global graph/manifold score plus train-only conditional target support."""

    def __init__(self, *args, train_frame: pd.DataFrame, conditional_weight: float,
                 conditional_neighbors: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_frame = train_frame.reset_index(drop=True)
        self.conditional_weight = float(conditional_weight)
        self.conditional_neighbors = int(conditional_neighbors)
        self.train_numeric = self.train_data.X[:, :self.train_data.d_numerical].cpu().numpy()
        self._context_cache = {}

    def _context_neighbors(self, base: pd.DataFrame, targets: tuple[str, ...]):
        key = tuple(targets)
        if key not in self._context_cache:
            positions = [i for i, name in enumerate(self.numeric_names) if name not in targets]
            if not positions:
                positions = list(range(len(self.numeric_names)))
            model = NearestNeighbors(
                n_neighbors=min(self.conditional_neighbors, len(self.train_frame))).fit(
                    self.train_numeric[:, positions])
            self._context_cache[key] = (positions, model)
        positions, model = self._context_cache[key]
        raw = base[self.numeric_names].apply(pd.to_numeric, errors="coerce")
        encoded = self.train_data.num_transform.transform(raw.fillna(raw.median()).to_numpy(float))
        return model.kneighbors(encoded[:, positions], return_distance=False)

    def conditional_support(self, base: pd.DataFrame, candidate: pd.DataFrame,
                            targets: tuple[str, ...]) -> np.ndarray:
        neighbors = self._context_neighbors(base, targets)
        terms = []
        for target in targets:
            if target in self.numeric_names:
                bank = pd.to_numeric(self.train_frame[target], errors="coerce").to_numpy(float)[neighbors]
                values = pd.to_numeric(candidate[target], errors="coerce").to_numpy(float)[:, None]
                global_scale = max(float(pd.to_numeric(
                    self.train_frame[target], errors="coerce").std()), 1e-6)
                median = np.nanmedian(bank, axis=1, keepdims=True)
                q25 = np.nanquantile(bank, .25, axis=1, keepdims=True)
                q75 = np.nanquantile(bank, .75, axis=1, keepdims=True)
                local_scale = np.maximum((q75 - q25) / 1.349, .05 * global_scale)
                nearest = np.nanmin(np.abs(bank - values), axis=1) / global_scale
                central = (np.abs(values - median) / local_scale).reshape(-1)
                terms.append(np.clip(.5 * nearest + .5 * central, 0, 20))
            elif target in self.categorical_names:
                bank = self.train_frame[target].astype(str).to_numpy()[neighbors]
                values = candidate[target].astype(str).to_numpy()
                cardinality = max(len(self.category_probability[target]), 1)
                count = (bank == values[:, None]).sum(axis=1)
                probability = (count + 1.0) / (bank.shape[1] + cardinality)
                terms.append(-np.log(np.maximum(probability, 1e-8)))
        return np.mean(terms, axis=0) if terms else np.zeros(len(base), dtype=float)

    def __call__(self, base: pd.DataFrame, candidate: pd.DataFrame,
                 targets: tuple[str, ...]) -> np.ndarray:
        return super().__call__(base, candidate, targets) + self.conditional_weight * self.conditional_support(
            base, candidate, targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-time", type=float, required=True)
    parser.add_argument("--trigger", choices=["mid", "final"], default="mid",
                        help="State used to activate the graph and condition proposal generation.")
    parser.add_argument("--proposal-source", choices=["trigger", "final"], default="trigger",
                        help="State conditioned on when drawing correction proposals.")
    parser.add_argument("--samples", type=int, default=2048); parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--branches", type=int, default=8); parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=3); parser.add_argument("--max-calibration-rate", type=float, default=.05)
    parser.add_argument("--graph-weight", type=float, default=20.0); parser.add_argument("--density-weight", type=float, default=1.0)
    parser.add_argument("--rarity-weight", type=float, default=.1); parser.add_argument("--edit-weight", type=float, default=.5)
    parser.add_argument("--conditional-weight", type=float, required=True)
    parser.add_argument("--conditional-neighbors", type=int, default=64)
    parser.add_argument("--accept-margin", type=float, default=0.0); parser.add_argument("--density-bank", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261440)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True); sample_dir = args.output / "samples"; sample_dir.mkdir()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")

    started = time.perf_counter(); device = torch.device("cuda")
    diffusion, train_data, test_data, info, _ = load_backbone(
        args.repo, args.dataset, args.config, args.checkpoint, device, args.steps)
    edges = json.loads(args.graph.read_text()); selected_edges = eligible_edges(edges, args.max_calibration_rate)
    if not selected_edges:
        raise RuntimeError("no graph edge passed the actuation/calibration filter")
    train = decode(train_data, train_data.X, info); heldout = decode(train_data, test_data.X, info)
    base_encoded, preview_encoded, actual_time, base_runtime = sample_base_and_preview(
        diffusion, args.samples, args.checkpoint_time, args.seed + 100000)
    base_frame = decode(train_data, base_encoded, info); preview_frame = decode(train_data, preview_encoded, info)
    base_frame.to_csv(sample_dir / "tabdiff.csv", index=False)
    preview_frame.to_csv(sample_dir / "trigger_preview.csv", index=False)
    trigger_encoded = preview_encoded if args.trigger == "mid" else base_encoded
    trigger_frame = preview_frame if args.trigger == "mid" else base_frame
    proposal_encoded = base_encoded if args.proposal_source == "final" else trigger_encoded
    scorer = ConditionalProposalScore(
        train_data, train, info, selected_edges, args.graph_weight, args.density_weight,
        args.rarity_weight, args.edit_weight, args.density_bank, train_frame=train,
        conditional_weight=args.conditional_weight, conditional_neighbors=args.conditional_neighbors)
    repaint, diagnostics = conditional_repaint(
        diffusion, train_data, info, base_frame, proposal_encoded, trigger_frame,
        selected_edges, scorer, args.branches, args.rounds, args.max_targets,
        args.seed + 200000, args.accept_margin)
    repaint.to_csv(sample_dir / "conditional_density_repaint.csv", index=False)
    actuator = TypedActuator(edges, train_data, train, info)
    base_repair, base_repair_diag = calibrated_minimal_final_repair(base_frame, actuator)
    repaint_repair, repaint_repair_diag = calibrated_minimal_final_repair(repaint, actuator)
    base_repair.to_csv(sample_dir / "calibrated_final_repair.csv", index=False)
    repaint_repair.to_csv(sample_dir / "conditional_density_repaint_plus_repair.csv", index=False)
    rows = [
        {"method": "tabdiff", **graph_metrics(base_frame, heldout, edges), "runtime_seconds": base_runtime},
        {"method": "calibrated_final_repair", **graph_metrics(base_repair, heldout, edges),
         "runtime_seconds": base_runtime, **base_repair_diag},
        {"method": "conditional_density_repaint", **graph_metrics(repaint, heldout, edges),
         "runtime_seconds": base_runtime + diagnostics["runtime_seconds"], **diagnostics},
        {"method": "conditional_density_repaint_plus_repair", **graph_metrics(repaint_repair, heldout, edges),
         "runtime_seconds": base_runtime + diagnostics["runtime_seconds"], **repaint_repair_diag},
    ]
    pd.DataFrame(rows).to_csv(args.output / "mechanism_metrics.csv", index=False)
    metadata = {
        "method": "conditional_density_graph_repainting", "dataset": args.dataset,
        "trigger": args.trigger,
        "proposal_source": args.proposal_source,
        "requested_checkpoint": args.checkpoint_time, "actual_checkpoint": actual_time,
        "graph_edges": len(edges), "eligible_edges": len(selected_edges), "api_calls": 0,
        "selection_uses_heldout": False, "diagnostics": diagnostics,
        "runtime_seconds": time.perf_counter() - started,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
