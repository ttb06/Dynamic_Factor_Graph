#!/usr/bin/env python3
"""Audit whether row-edge state changes survive and agree with the final sample.

The probe does not guide sampling.  It observes an unconditional TabDiff
trajectory and labels each applicability/violation flip by whether the new
state agrees with the final generated row.  Waiting for k consecutive
observations is evaluated as a transition certifier with k-1 steps of delay.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditioned_tabdiff_adapter import decode, seed_all
from subgroup_trajectory_probe import TracePreviewModel, time_bin
from tabdiff_probe import preprocess_dataset
from typed_dynamic_graph_guidance import edge_masks


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def evaluate_wait_policy(
    states: np.ndarray,
    final_state: np.ndarray,
    times: np.ndarray,
    candidate_id: str,
    state_kind: str,
    max_wait: int,
    active_min_t: float,
    active_max_t: float,
) -> list[dict]:
    """Evaluate delayed acceptance of raw flips.

    A flip at index j is certified by wait=k when the new state observed at j
    is unchanged for k observations (j through j+k-1).  The decision therefore
    occurs at j+k-1.  The final generated row is used only as an evaluation
    label, never as an input to the certifier.
    """
    rows: list[dict] = []
    for index in range(1, len(times)):
        t = float(times[index])
        if not active_min_t <= t <= active_max_t:
            continue
        flipped = states[index] != states[index - 1]
        if not flipped.any():
            continue
        new_state = states[index]
        correct = new_state == final_state
        direction = np.where(new_state, "on", "off")
        for wait in range(1, max_wait + 1):
            decision_index = index + wait - 1
            if decision_index >= len(times):
                continue
            survived = flipped.copy()
            for future in range(index + 1, decision_index + 1):
                survived &= states[future] == new_state
            for transition_direction in ("on", "off", "all"):
                eligible = flipped if transition_direction == "all" else flipped & (direction == transition_direction)
                accepted = survived if transition_direction == "all" else survived & (direction == transition_direction)
                raw_correct = eligible & correct
                accepted_correct = accepted & correct
                rows.append({
                    "candidate_id": candidate_id,
                    "state_kind": state_kind,
                    "source_index": index,
                    "source_t": t,
                    "source_time_bin": time_bin(t),
                    "decision_index": decision_index,
                    "decision_t": float(times[decision_index]),
                    "wait": wait,
                    "direction": transition_direction,
                    "raw_flips": int(eligible.sum()),
                    "raw_final_correct": int(raw_correct.sum()),
                    "accepted": int(accepted.sum()),
                    "accepted_final_correct": int(accepted_correct.sum()),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-wait", type=int, default=5)
    parser.add_argument("--active-min-t", type=float, default=.10)
    parser.add_argument("--active-max-t", type=float, default=.75)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")

    edges = [
        edge for edge in json.loads(args.graph.read_text())
        if edge.get("usage_role", "actuate") == "actuate"
    ]
    if not edges:
        raise RuntimeError("no actuation edges")

    seed_all(args.seed)
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train_data, _ = preprocess_dataset(args.repo, args.dataset)
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    base = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(checkpoint["denoise_fn"], strict=True)
    traced = TracePreviewModel(base, train_data.categories).cuda()
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
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories,
        train_data.d_numerical,
        traced,
        None,
        **diffusion_params,
        device=torch.device("cuda"),
    ).cuda()
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    diffusion.eval()

    traced.start()
    with torch.no_grad():
        final_encoded = diffusion.sample(args.samples)
    history = traced.stop()
    final = decode(train_data, final_encoded, info)
    times = np.asarray([float(trace["t"]) for trace in history], dtype=float)

    state_store: dict[tuple[str, str], list[np.ndarray]] = {
        (edge["candidate_id"], kind): []
        for edge in edges
        for kind in ("applicability", "violation")
    }
    final_store: dict[tuple[str, str], np.ndarray] = {}
    for edge in edges:
        active, _, violation = edge_masks(final, edge)
        final_store[(edge["candidate_id"], "applicability")] = active
        final_store[(edge["candidate_id"], "violation")] = violation

    for trace in history:
        encoded = torch.cat([trace["numeric"], trace["codes"].float()], dim=1)
        preview = decode(train_data, encoded, info)
        for edge in edges:
            active, _, violation = edge_masks(preview, edge)
            edge_id = edge["candidate_id"]
            state_store[(edge_id, "applicability")].append(active)
            state_store[(edge_id, "violation")].append(violation)

    event_rows: list[dict] = []
    for edge in edges:
        edge_id = edge["candidate_id"]
        for state_kind in ("applicability", "violation"):
            states = np.stack(state_store[(edge_id, state_kind)], axis=0)
            event_rows.extend(evaluate_wait_policy(
                states,
                final_store[(edge_id, state_kind)],
                times,
                edge_id,
                state_kind,
                args.max_wait,
                args.active_min_t,
                args.active_max_t,
            ))

    events = pd.DataFrame(event_rows)
    if events.empty:
        raise RuntimeError("no transition events in active window")
    events["dataset"] = args.dataset
    events["seed"] = args.seed
    events["raw_precision"] = events.apply(
        lambda row: safe_ratio(int(row.raw_final_correct), int(row.raw_flips)), axis=1
    )
    events["certified_precision"] = events.apply(
        lambda row: safe_ratio(int(row.accepted_final_correct), int(row.accepted)), axis=1
    )
    events.to_csv(args.output / "transition_events.csv", index=False)

    grouped = events.groupby(["state_kind", "source_time_bin", "wait", "direction"], sort=False).agg(
        raw_flips=("raw_flips", "sum"),
        raw_final_correct=("raw_final_correct", "sum"),
        accepted=("accepted", "sum"),
        accepted_final_correct=("accepted_final_correct", "sum"),
    ).reset_index()
    grouped["raw_precision"] = grouped.raw_final_correct / grouped.raw_flips.replace(0, np.nan)
    grouped["certified_precision"] = grouped.accepted_final_correct / grouped.accepted.replace(0, np.nan)
    grouped["genuine_event_recall"] = grouped.accepted_final_correct / grouped.raw_final_correct.replace(0, np.nan)
    grouped["acceptance_rate"] = grouped.accepted / grouped.raw_flips.replace(0, np.nan)
    grouped.to_csv(args.output / "transition_policy_summary.csv", index=False)

    main_table = grouped[(grouped.direction == "all") & grouped.source_time_bin.isin(["(0.50,0.75]", "(0.25,0.50]", "(0.10,0.25]"])]
    lines = [
        "# Transition survival probe",
        "",
        f"Dataset `{args.dataset}`, seed `{args.seed}`, rows `{args.samples}`, actuation edges `{len(edges)}`.",
        "",
        "A certified transition waits for `k` identical observations. Final rows are evaluation labels only.",
        "",
        "| State | Source bin | Wait | Raw flips | Accepted | Certified precision | Genuine-event recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in main_table.itertuples(index=False):
        lines.append(
            f"| {row.state_kind} | {row.source_time_bin} | {row.wait} | {row.raw_flips} | "
            f"{row.accepted} | {row.certified_precision:.4f} | {row.genuine_event_recall:.4f} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "dataset": args.dataset,
        "seed": args.seed,
        "samples": args.samples,
        "steps": args.steps,
        "actuation_edges": len(edges),
        "max_wait": args.max_wait,
        "active_window": [args.active_min_t, args.active_max_t],
        "uses_final_state_as_input": False,
        "api_calls": 0,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
