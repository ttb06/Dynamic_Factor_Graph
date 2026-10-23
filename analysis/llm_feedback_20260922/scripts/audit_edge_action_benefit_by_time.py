#!/usr/bin/env python3
"""Train-only audit of whether each typed graph action helps x0 recovery over time.

For a clean training row we know x0.  We forward-noise it at fixed diffusion
times, decode the frozen model's x0 estimate, execute one graph edge at a time,
and measure the change in squared reconstruction error.  This is an analysis
and calibration instrument: it never reads held-out rows or final sample
metrics and never calls an LLM.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from conditioned_tabdiff_adapter import decode
from dense_numeric_factor_messages import DenseNumericFactors
from finetune_trajectory_teacher_tabdiff import (
    categorical_codes,
    noisy_view,
    seed_all,
)
from tabdiff_probe import preprocess_dataset


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return float("nan")
    probability = successes / total
    denominator = 1.0 + z * z / total
    centre = probability + z * z / (2.0 * total)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return (centre - radius) / denominator


def edge_target(frame: pd.DataFrame, teacher_num: torch.Tensor,
                factors: DenseNumericFactors, edge: dict, strength: float,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    original_edges = factors.edges
    factors.edges = [edge]
    try:
        corrected, changed, events = factors.project_violations(frame, strength)
    finally:
        factors.edges = original_edges
    target = torch.as_tensor(
        factors.encode_numeric(corrected), dtype=teacher_num.dtype, device=device)
    mask = torch.as_tensor(changed, dtype=torch.bool, device=device)
    return target, mask, int(events)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="news")
    parser.add_argument("--regression-target-mode", choices=["categorical", "numerical"],
                        default="numerical")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--times", default=".95,.75,.55,.40,.30,.20,.12,.07,.03")
    parser.add_argument("--strength", type=float, default=.50)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    seed_all(args.seed)
    device = torch.device("cuda")
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    times = [float(value) for value in args.times.split(",") if value.strip()]
    train_data, _ = preprocess_dataset(
        args.repo, args.dataset, regression_target_mode=args.regression_target_mode)
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    raw_train = decode(train_data, train_data.X, info)
    edges = copy.deepcopy(json.loads(args.graph.read_text()))
    factors = DenseNumericFactors(
        train_data, raw_train, info, edges, args.seed, trees=0, min_r2=0.0)

    with args.config.open("rb") as handle:
        config = copy.deepcopy(pickle.load(handle))
    parameters = copy.deepcopy(config["unimodmlp_params"])
    parameters["d_numerical"] = train_data.d_numerical
    parameters["categories"] = (train_data.categories + 1).tolist()
    model = Model(UniModMLP(**parameters), **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["denoise_fn"], strict=True)
    diffusion_parameters = copy.deepcopy(config["diffusion_params"])
    diffusion_parameters.update(
        scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, model, None,
        **diffusion_parameters, device=device).to(device).eval()
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    teacher = diffusion._denoise_fn.eval()

    selected = train_data.X[: min(args.rows, len(train_data.X))]
    loader = DataLoader(
        TensorDataset(selected), batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=0)
    statistics: dict[tuple[float, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float))
    gate_examples: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    row_offset = 0

    for time_value in times:
        # The same deterministic noise stream is shared across edges at a time.
        seed_all(args.seed + int(round(time_value * 10000)))
        row_offset = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.no_grad():
                view = noisy_view(diffusion, batch, time_value, time_value, teacher)
                sizes = [int(value) for value in train_data.categories]
                codes = categorical_codes(view["noisy_cat"], view["teacher_logits"], sizes)
                encoded = torch.cat(
                    [view["teacher_num"], codes.to(view["teacher_num"].dtype)], dim=1)
                frame = decode(train_data, encoded.cpu(), info)
                clean = view["clean_num"]
                teacher_num = view["teacher_num"]
                sigma_num = diffusion.num_schedule.total_noise(
                    torch.full((len(batch), 1), time_value,
                               dtype=teacher_num.dtype, device=device))
                for edge in edges:
                    target, mask, events = edge_target(
                        frame, teacher_num, factors, edge, args.strength, device)
                    key = (time_value, edge["candidate_id"])
                    summary = statistics[key]
                    changed = int(mask.sum())
                    summary["events"] += events
                    summary["changed_cells"] += changed
                    summary["rows"] += len(batch)
                    if changed == 0:
                        continue
                    benefit = (teacher_num - clean).square() - (target - clean).square()
                    changed_benefit = benefit[mask]
                    row_benefit = (benefit * mask.to(benefit.dtype)).sum(dim=1)
                    changed_rows = mask.any(dim=1)
                    changed_row_benefit = row_benefit[changed_rows]
                    mask_float = mask.to(teacher_num.dtype)
                    changed_count = mask_float.sum(dim=1).clamp_min(1.0)
                    delta = target - teacher_num
                    features = torch.stack([
                        torch.full_like(changed_count, time_value),
                        (delta.abs() * mask_float).sum(dim=1) / changed_count,
                        ((delta.square() * mask_float).sum(dim=1) / changed_count).sqrt(),
                        (teacher_num.abs() * mask_float).sum(dim=1) / changed_count,
                        (sigma_num * mask_float).sum(dim=1) / changed_count,
                        teacher_num.abs().mean(dim=1),
                        teacher_num.std(dim=1),
                    ], dim=1)
                    local_rows = torch.arange(
                        row_offset, row_offset + len(batch), device=device)[changed_rows]
                    gate_examples[edge["candidate_id"]].append({
                        "x": features[changed_rows].cpu().numpy(),
                        "label": (changed_row_benefit > 0).cpu().numpy().astype(np.int8),
                        "benefit": changed_row_benefit.cpu().numpy(),
                        "row_id": local_rows.cpu().numpy(),
                    })
                    summary["changed_rows"] += int(changed_rows.sum())
                    summary["beneficial_cells"] += int((changed_benefit > 0).sum())
                    summary["beneficial_rows"] += int((changed_row_benefit > 0).sum())
                    summary["net_benefit"] += float(changed_benefit.sum())
                    summary["absolute_action"] += float(
                        (target - teacher_num).abs()[mask].sum())
            row_offset += len(batch)

    records = []
    by_id = {edge["candidate_id"]: edge for edge in edges}
    for (time_value, edge_id), summary in sorted(
            statistics.items(), key=lambda item: (-item[0][0], item[0][1])):
        cells = int(summary["changed_cells"])
        rows = int(summary["changed_rows"])
        positive_cells = int(summary["beneficial_cells"])
        positive_rows = int(summary["beneficial_rows"])
        edge = by_id[edge_id]
        records.append({
            "t": time_value,
            "edge_id": edge_id,
            "relation_family": edge.get("relation_family", ""),
            "description": edge.get("description", ""),
            "events": int(summary["events"]),
            "changed_cells": cells,
            "changed_rows": rows,
            "beneficial_cell_fraction": positive_cells / max(cells, 1),
            "beneficial_cell_wilson_lower": wilson_lower(positive_cells, cells),
            "beneficial_row_fraction": positive_rows / max(rows, 1),
            "beneficial_row_wilson_lower": wilson_lower(positive_rows, rows),
            "mean_net_benefit_per_changed_cell": summary["net_benefit"] / max(cells, 1),
            "mean_absolute_action": summary["absolute_action"] / max(cells, 1),
        })
    result = pd.DataFrame(records)
    result.to_csv(args.output / "edge_time_benefit.csv", index=False)

    feature_names = [
        "t", "mean_abs_action", "rms_action", "changed_preview_abs",
        "changed_sigma", "global_preview_abs", "global_preview_std",
    ]
    gate_directory = args.output / "benefit_gate"
    gate_directory.mkdir()
    gate_records = []
    gate_manifest = {
        "feature_names": feature_names,
        "proposal_strength": args.strength,
        "models": {},
    }
    for edge_id, chunks in sorted(gate_examples.items()):
        if not chunks:
            continue
        x = np.concatenate([chunk["x"] for chunk in chunks])
        label = np.concatenate([chunk["label"] for chunk in chunks])
        benefit = np.concatenate([chunk["benefit"] for chunk in chunks])
        row_id = np.concatenate([chunk["row_id"] for chunk in chunks])
        fit = row_id % 5 != 0
        validation = ~fit
        if fit.sum() < 100 or validation.sum() < 20 or len(np.unique(label[fit])) < 2:
            gate_records.append({
                "edge_id": edge_id, "examples": len(label), "status": "insufficient_labels"})
            continue
        positive = max(int(label[fit].sum()), 1)
        negative = max(int(fit.sum() - positive), 1)
        weights = np.where(
            label[fit] > 0, fit.sum() / (2.0 * positive), fit.sum() / (2.0 * negative))
        model = HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=120, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0, random_state=args.seed)
        model.fit(x[fit], label[fit], sample_weight=weights)
        probability = model.predict_proba(x[validation])[:, 1]
        candidates = np.unique(np.concatenate([
            np.linspace(.05, .95, 19), np.quantile(probability, np.linspace(.05, .95, 19))
        ]))
        best_threshold = 1.0
        best_total_benefit = 0.0
        for threshold in candidates:
            selected_mask = probability >= threshold
            total_benefit = float(benefit[validation][selected_mask].sum())
            if total_benefit > best_total_benefit:
                best_total_benefit = total_benefit
                best_threshold = float(threshold)
        selected_mask = probability >= best_threshold
        true_positive = selected_mask & (label[validation] > 0)
        selected_count = int(selected_mask.sum())
        validation_positive = int(label[validation].sum())
        model_path = gate_directory / f"{edge_id}.joblib"
        joblib.dump(model, model_path)
        gate_manifest["models"][edge_id] = {
            "path": model_path.name,
            "threshold": best_threshold,
        }
        gate_records.append({
            "edge_id": edge_id,
            "examples": len(label),
            "train_examples": int(fit.sum()),
            "validation_examples": int(validation.sum()),
            "positive_rate": float(label[validation].mean()),
            "average_precision": float(average_precision_score(label[validation], probability)),
            "roc_auc": float(roc_auc_score(label[validation], probability)),
            "threshold": best_threshold,
            "selected_fraction": selected_count / max(int(validation.sum()), 1),
            "selected_precision": int(true_positive.sum()) / max(selected_count, 1),
            "selected_recall": int(true_positive.sum()) / max(validation_positive, 1),
            "selected_total_benefit": best_total_benefit,
            "all_action_total_benefit": float(benefit[validation].sum()),
            "status": "trained",
        })
    pd.DataFrame(gate_records).to_csv(gate_directory / "validation_metrics.csv", index=False)
    (gate_directory / "manifest.json").write_text(json.dumps(gate_manifest, indent=2))

    policy = {}
    for edge_id, group in result.groupby("edge_id", sort=True):
        useful = group[
            (group["changed_cells"] >= 50)
            & (group["mean_net_benefit_per_changed_cell"] > 0)
            & (group["beneficial_cell_wilson_lower"] > .50)
        ]
        policy[edge_id] = {
            "eligible_times": useful["t"].astype(float).tolist(),
            "best_time": (None if useful.empty else float(
                useful.sort_values("mean_net_benefit_per_changed_cell", ascending=False).iloc[0]["t"])),
            "criterion": "train-only positive mean benefit and Wilson lower bound > 0.5",
        }
    (args.output / "candidate_time_policy.json").write_text(json.dumps(policy, indent=2))
    metadata = {
        "dataset": args.dataset,
        "rows": len(selected),
        "times": times,
        "edges": len(edges),
        "strength": args.strength,
        "heldout_used": False,
        "final_metrics_used": False,
        "seed": args.seed,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
