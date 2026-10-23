#!/usr/bin/env python3
"""General state-conditioned constraint-graph probe for TabDiff trajectories.

The specification graph is discovered from the training split and represented
as typed implications.  At every predicted-clean state, this script measures
(1) row-level edge instantiation and (2) batch-level edge reliability.  It does
not enforce constraints and never emits real source rows.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditioned_tabdiff_adapter import decode, seed_all
from subgroup_trajectory_probe import TracePreviewModel, time_bin
from tabdiff_probe import preprocess_dataset


IMPLICATION = re.compile(r"^\s*IF\s+(.*?)\s+THEN\s+(.*?)\s*$", re.I)
PREDICATE = re.compile(r"^\s*(.*?)\s*(==|>=|<=|>|<)\s*(.*?)\s*$")


def parse_implication(text: str) -> tuple[str, str]:
    match = IMPLICATION.match(str(text))
    if not match:
        raise ValueError(text)
    return tuple(value.strip() for value in match.groups())


def evaluate(frame: pd.DataFrame, expression: str) -> np.ndarray:
    match = PREDICATE.match(expression)
    if not match:
        raise ValueError(expression)
    column, operator, literal = (value.strip() for value in match.groups())
    if column not in frame or "<MISSING_LIKE>" in literal:
        raise ValueError(expression)
    values = frame[column]
    if operator == "==":
        numeric = pd.to_numeric(values, errors="coerce")
        try:
            threshold = float(literal)
            if numeric.notna().mean() > 0.95:
                return np.isclose(numeric.to_numpy(float), threshold, rtol=0, atol=1e-7)
        except ValueError:
            pass
        return values.astype(str).str.strip().eq(literal).to_numpy()
    numeric = pd.to_numeric(values, errors="coerce")
    threshold = float(literal)
    functions = {">": numeric.gt, "<": numeric.lt, ">=": numeric.ge, "<=": numeric.le}
    return functions[operator](threshold).fillna(False).to_numpy()


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def confusion(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    tp = int((predicted & reference).sum())
    fp = int((predicted & ~reference).sum())
    fn = int((~predicted & reference).sum())
    tn = int((~predicted & ~reference).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
    }


def select_candidates(catalog: pd.DataFrame, dataset: str, maximum: int, min_support: int) -> pd.DataFrame:
    part = catalog[
        catalog.dataset.eq(dataset)
        & catalog.source.eq("train_only_predicate_mining")
        & ~catalog.sensitive_attribute.astype(bool)
        & ~catalog.target_related.astype(bool)
        & catalog.train_applicable.ge(min_support)
    ].copy()
    part = part[~part.constraint.astype(str).str.contains("<MISSING_LIKE>", regex=False)]
    valid = []
    for row in part.itertuples(index=False):
        try:
            left, right = parse_implication(row.constraint)
            PREDICATE.match(left).groups()
            PREDICATE.match(right).groups()
            valid.append(True)
        except Exception:
            valid.append(False)
    part = part[np.asarray(valid, dtype=bool)].copy()
    part["split_stability"] = 1.0 - (part.train_violation_rate - part.test_violation_rate).abs().clip(0, 1)
    part["priority"] = (
        np.log1p(part.train_applicable)
        * np.log1p(part.train_lift.clip(lower=0))
        * part.split_stability
    )
    # Preserve relation-family and endpoint diversity instead of selecting many
    # threshold variants of the same two columns.
    selected = []
    seen_pairs: dict[tuple[str, str], int] = {}
    category_order = [
        "executable_baseline_candidate", "near_hard_dynamic_benchmark",
        "probabilistic_dependency_not_hard_validity", "subgroup_or_shift_stress_test",
        "insufficient_support_stress_test",
    ]
    ordered = part.sort_values("priority", ascending=False)
    for category in category_order:
        block = ordered[ordered.use_category.eq(category)]
        quota = max(2, maximum // max(len(category_order), 1))
        added = 0
        for _, row in block.iterrows():
            pair = (str(row.antecedent_column), str(row.consequent_column))
            if seen_pairs.get(pair, 0) >= 2:
                continue
            selected.append(row)
            seen_pairs[pair] = seen_pairs.get(pair, 0) + 1
            added += 1
            if len(selected) >= maximum or added >= quota:
                break
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        keys = {str(row.constraint_id) for row in selected}
        for _, row in ordered.iterrows():
            if str(row.constraint_id) in keys:
                continue
            selected.append(row)
            if len(selected) >= maximum:
                break
    return pd.DataFrame(selected).reset_index(drop=True)


def state_stats(antecedent: np.ndarray, consequent: np.ndarray) -> dict[str, float]:
    support = int(antecedent.sum())
    violation = int((antecedent & ~consequent).sum())
    confidence = ratio(int((antecedent & consequent).sum()), support)
    base = float(consequent.mean())
    lift = confidence / base if base > 0 and np.isfinite(confidence) else math.nan
    return {
        "support": support, "violation": violation,
        "violation_rate": ratio(violation, support),
        "confidence": confidence, "base_rate": base, "lift": lift,
    }


def summarize(rows: pd.DataFrame, output: Path, min_batch_support: int) -> None:
    configs = [
        ("hard", 0.99, 1.05),
        ("near_hard", 0.90, 1.10),
        ("dependency", 0.70, 1.25),
    ]
    summaries = []
    status_rows = []
    for name, confidence_cut, lift_cut in configs:
        working = rows.copy()
        working["enabled"] = (
            working.support.ge(min_batch_support)
            & working.confidence.ge(confidence_cut)
            & working.lift.ge(lift_cut)
        )
        final = working.sort_values("call_index").groupby("candidate_id").tail(1)[
            ["candidate_id", "enabled"]
        ].rename(columns={"enabled": "final_enabled"})
        working = working.merge(final, on="candidate_id", validate="many_to_one")
        # Persistence is evaluated along the reverse trajectory (call_index grows
        # from noisy to clean). Two consecutive enabled states are required.
        working = working.sort_values(["candidate_id", "call_index"])
        previous = working.groupby("candidate_id").enabled.shift(1).fillna(False).astype(bool)
        working["persistent_enabled"] = working.enabled & previous
        for gate in ["enabled", "persistent_enabled"]:
            for time_name, block in working.groupby("time_bin", sort=False):
                metrics = confusion(block[gate].to_numpy(bool), block.final_enabled.to_numpy(bool))
                summaries.append({
                    "profile": name, "gate": gate, "time_bin": time_name,
                    "edges": len(block), **metrics,
                })
        working["profile"] = name
        status_rows.append(working)
    status = pd.concat(status_rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    status.to_csv(output / "state_graph_status.csv", index=False)
    summary.to_csv(output / "state_graph_summary.csv", index=False)

    row_summary = rows.groupby("time_bin", sort=False).agg(
        active_precision=("active_precision", "mean"),
        active_recall=("active_recall", "mean"),
        violation_precision=("violation_precision", "mean"),
        violation_recall=("violation_recall", "mean"),
        topology_flip=("active_flip", "mean"),
        evaluated_edges=("candidate_id", "nunique"),
    ).reset_index()
    row_summary.to_csv(output / "row_graph_summary.csv", index=False)

    order = ["(0.75,1.0]", "(0.50,0.75]", "(0.25,0.50]", "(0.10,0.25]", "[0,0.10]"]
    row_summary["order"] = row_summary.time_bin.map({value: index for index, value in enumerate(order)})
    row_summary = row_summary.sort_values("order")
    lines = [
        "# General state-conditioned graph probe", "",
        "The specification graph was discovered from training data. Intermediate clean-record estimates instantiate row-specific edges and re-estimate batch-level edge reliability. Final generated states are used only as trajectory reference labels.",
        "", "## Row-level graph readability", "",
        "| Time bin | Active P/R | Violation P/R | Topology flip | Edges |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in row_summary.itertuples(index=False):
        lines.append(
            f"| {row.time_bin} | {row.active_precision:.3f}/{row.active_recall:.3f} | "
            f"{row.violation_precision:.3f}/{row.violation_recall:.3f} | {row.topology_flip:.3f} | {int(row.evaluated_edges)} |"
        )
    lines += ["", "## Batch-level discovered graph", "",
              "| Profile | Gate | Time bin | Precision | Recall |", "| --- | --- | --- | ---: | ---: |"]
    summary["order"] = summary.time_bin.map({value: index for index, value in enumerate(order)})
    for row in summary.sort_values(["profile", "gate", "order"]).itertuples(index=False):
        lines.append(f"| {row.profile} | {row.gate} | {row.time_bin} | {row.precision:.3f} | {row.recall:.3f} |")
    lines += [
        "", "## Interpretation boundary", "",
        "- Row-level active means that the implication antecedent holds in the current state; violation means its consequent does not hold.",
        "- Batch-level enabled means that current synthetic states provide enough support, confidence, and lift for the edge under the named profile.",
        "- These are trajectory-readability measurements, not proof that a discovered relation is a semantic invariant.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-edges", type=int, default=24)
    parser.add_argument("--min-train-support", type=int, default=100)
    parser.add_argument("--min-batch-support", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    seed_all(args.seed)
    catalog = pd.read_csv(args.catalog)
    candidates = select_candidates(catalog, args.dataset, args.max_edges, args.min_train_support)
    if candidates.empty:
        raise RuntimeError("no candidate implications selected")
    candidates.to_csv(args.output / "selected_specification_graph.csv", index=False)

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
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(
        train_data.categories, train_data.d_numerical, traced, None,
        **diffusion_params, device=torch.device("cuda"),
    ).cuda()
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    diffusion.eval()
    traced.start()
    with torch.no_grad():
        final_encoded = diffusion.sample(args.samples)
    history = traced.stop()
    final = decode(train_data, final_encoded, info)

    final_masks = {}
    for candidate in candidates.itertuples(index=False):
        left, right = parse_implication(candidate.constraint)
        antecedent = evaluate(final, left)
        consequent = evaluate(final, right)
        final_masks[candidate.constraint_id] = (antecedent, consequent)

    rows = []
    previous_active = {}
    for call_index, trace in enumerate(history):
        encoded = torch.cat([trace["numeric"], trace["codes"].float()], dim=1)
        preview = decode(train_data, encoded, info)
        for candidate in candidates.itertuples(index=False):
            left, right = parse_implication(candidate.constraint)
            antecedent = evaluate(preview, left)
            consequent = evaluate(preview, right)
            violation = antecedent & ~consequent
            final_antecedent, final_consequent = final_masks[candidate.constraint_id]
            final_violation = final_antecedent & ~final_consequent
            active_metrics = confusion(antecedent, final_antecedent)
            violation_metrics = confusion(violation, final_violation)
            previous = previous_active.get(candidate.constraint_id)
            flip = float((antecedent != previous).mean()) if previous is not None else math.nan
            previous_active[candidate.constraint_id] = antecedent
            stats = state_stats(antecedent, consequent)
            rows.append({
                "dataset": args.dataset, "candidate_id": candidate.constraint_id,
                "constraint": candidate.constraint, "family": candidate.family,
                "use_category": candidate.use_category, "call_index": call_index,
                "t": trace["t"], "time_bin": time_bin(trace["t"]),
                **stats,
                "active_precision": active_metrics["precision"], "active_recall": active_metrics["recall"],
                "active_accuracy": active_metrics["accuracy"],
                "violation_precision": violation_metrics["precision"], "violation_recall": violation_metrics["recall"],
                "violation_accuracy": violation_metrics["accuracy"], "active_flip": flip,
            })
    trajectory = pd.DataFrame(rows)
    trajectory.to_csv(args.output / "edge_state_trajectory.csv", index=False)
    summarize(trajectory, args.output, args.min_batch_support)
    (args.output / "run_metadata.json").write_text(json.dumps({
        "dataset": args.dataset, "samples": args.samples, "steps": args.steps,
        "selected_edges": len(candidates), "seed": args.seed,
        "catalog": str(args.catalog), "checkpoint": str(args.checkpoint),
        "real_rows_emitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
