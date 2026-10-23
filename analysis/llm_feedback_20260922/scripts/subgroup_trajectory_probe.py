#!/usr/bin/env python3
"""Probe when subgroup and subgroup-edge scope become readable in x-hat-0."""
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
from tabdiff_probe import preprocess_dataset


PREDICATE = re.compile(r"^\s*(.*?)\s*(==|>=|<=|>|<)\s*(.*?)\s*$")


def parse_predicate(text: str) -> tuple[str, str, str]:
    match = PREDICATE.match(text)
    if not match:
        raise ValueError(f"unsupported predicate: {text}")
    return tuple(part.strip() for part in match.groups())


def evaluate(frame: pd.DataFrame, description: str) -> np.ndarray:
    column, operator, literal = parse_predicate(description)
    if column not in frame:
        raise KeyError(column)
    if operator == "==":
        return frame[column].astype(str).str.strip().eq(literal).to_numpy()
    values = pd.to_numeric(frame[column], errors="coerce")
    threshold = float(literal)
    functions = {">": values.gt, "<": values.lt, ">=": values.ge, "<=": values.le}
    return functions[operator](threshold).fillna(False).to_numpy()


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def time_bin(t: float) -> str:
    if t > 0.75: return "(0.75,1.0]"
    if t > 0.50: return "(0.50,0.75]"
    if t > 0.25: return "(0.25,0.50]"
    if t > 0.10: return "(0.10,0.25]"
    return "[0,0.10]"


class TracePreviewModel(torch.nn.Module):
    def __init__(self, base, category_sizes):
        super().__init__()
        self.base = base
        self.category_sizes = [int(value) for value in category_sizes]
        self.enabled = False
        self.history = []

    def start(self):
        self.history = []
        self.enabled = True

    def stop(self):
        self.enabled = False
        return self.history

    def predicted_codes(self, x_cat, logits):
        current_chunks = torch.split(x_cat, [size + 1 for size in self.category_sizes], dim=-1)
        logit_chunks = torch.split(logits, [size + 1 for size in self.category_sizes], dim=-1)
        codes = []
        for size, current, prediction in zip(self.category_sizes, current_chunks, logit_chunks):
            current_code = current.argmax(dim=-1)
            predicted = prediction[:, :size].argmax(dim=-1)
            codes.append(torch.where(current_code.eq(size), predicted, current_code))
        return torch.stack(codes, dim=1)

    def forward(self, x_num, x_cat, t, sigma=None):
        output = self.base(x_num, x_cat, t, sigma=sigma)
        if self.enabled:
            self.history.append({
                "t": float(t.detach().float().mean().cpu()),
                "numeric": output[0].detach().cpu(),
                "codes": self.predicted_codes(x_cat, output[1]).detach().cpu(),
            })
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    seed_all(args.seed)
    device = torch.device("cuda")
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    numeric_names = {info["column_names"][index] for index in info["num_col_idx"]}
    candidates = pd.read_csv(args.audit / "all_subgroup_edge_candidates.csv")
    candidates = candidates[
        candidates.dataset.eq(args.dataset) & candidates.replicated.astype(bool)
        & ~candidates.sensitive_or_demographic.astype(bool)
    ].copy()
    # Exact equality on a continuous column is not a robust sampler predicate:
    # real data can have a zero point mass while continuous diffusion almost
    # never decodes exactly zero. Keep categorical equality, inequalities and
    # tail predicates in this trajectory screen.
    def robust_subgroup(description: str) -> bool:
        column, operator, _ = parse_predicate(str(description))
        return not (column in numeric_names and operator == "==")
    candidates = candidates[candidates.subgroup.map(robust_subgroup)]
    candidates["priority"] = candidates.test_interaction.abs()
    candidates = candidates.sort_values("priority", ascending=False).drop_duplicates(
        ["constraint_id", "subgroup_id"]
    ).head(args.max_candidates)
    if candidates.empty:
        raise RuntimeError("no replicated candidates for trajectory probe")

    train_data, _ = preprocess_dataset(args.repo, args.dataset)
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    base = Model(__import__("tabdiff.modules.main_modules", fromlist=["UniModMLP"]).UniModMLP(**params),
                 **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(checkpoint["denoise_fn"], strict=True)
    traced = TracePreviewModel(base, train_data.categories).to(device)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, traced, None,
                                      **diffusion_params, device=device).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    diffusion.eval()
    traced.start()
    with torch.no_grad():
        final_encoded = diffusion.sample(args.samples)
    history = traced.stop()
    final = decode(train_data, final_encoded, info)
    rows = []
    # Synthetic-row masks are retained in memory only. They let us test whether
    # temporal persistence improves active-graph decisions over a one-shot gate.
    activation_states = {}
    for call_index, trace in enumerate(history):
        encoded = torch.cat([trace["numeric"], trace["codes"].float()], dim=1)
        preview = decode(train_data, encoded, info)
        for candidate in candidates.itertuples(index=False):
            antecedent = str(candidate.constraint).split("IF ", 1)[1].split(" THEN ", 1)[0]
            for scope_name, predicate_list in [
                ("subgroup", [candidate.subgroup]),
                ("edge_scope", [antecedent, candidate.subgroup]),
            ]:
                predicted = np.ones(len(preview), dtype=bool)
                final_label = np.ones(len(final), dtype=bool)
                for predicate in predicate_list:
                    predicted &= evaluate(preview, predicate)
                    final_label &= evaluate(final, predicate)
                key = (candidate.candidate_id, scope_name)
                state = activation_states.setdefault(key, {
                    "constraint": candidate.constraint,
                    "subgroup": candidate.subgroup,
                    "scope": scope_name,
                    "final": final_label,
                    "predictions": [],
                })
                state["predictions"].append(predicted)
                tp = int((predicted & final_label).sum())
                tn = int((~predicted & ~final_label).sum())
                evaluable = int(final_label.sum()) >= 5
                rows.append({
                    "candidate_id": candidate.candidate_id,
                    "constraint": candidate.constraint,
                    "subgroup": candidate.subgroup,
                    "scope": scope_name,
                    "call_index": call_index,
                    "t": trace["t"],
                    "time_bin": time_bin(trace["t"]),
                    "final_positive_rate": float(final_label.mean()),
                    "predicted_positive_rate": float(predicted.mean()),
                    "accuracy_to_final": float((predicted == final_label).mean()),
                    "precision_to_final": ratio(tp, int(predicted.sum())) if evaluable else math.nan,
                    "recall_to_final": ratio(tp, int(final_label.sum())) if evaluable else math.nan,
                    "specificity_to_final": ratio(tn, int((~final_label).sum())),
                    "evaluable": evaluable,
                })
    metrics = pd.DataFrame(rows)
    metrics["flip_from_previous"] = metrics.groupby(["candidate_id", "scope"], sort=False).predicted_positive_rate.diff().abs()
    metrics.to_csv(args.output / "subgroup_trajectory.csv", index=False)
    summary = metrics.groupby(["scope", "time_bin"], sort=False).agg(
        candidates=("candidate_id", "nunique"), accuracy=("accuracy_to_final", "mean"),
        evaluable_candidates=("candidate_id", lambda values: values[metrics.loc[values.index, "evaluable"]].nunique()),
        precision=("precision_to_final", "mean"), recall=("recall_to_final", "mean"),
        positive_rate=("final_positive_rate", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary_by_time_bin.csv", index=False)
    schedule_rows = []
    for (candidate_id, scope_name), state in activation_states.items():
        predictions = state["predictions"]
        final_label = state["final"]
        for index, single in enumerate(predictions):
            windows = {"single": single}
            if index >= 1:
                windows["consecutive_2"] = single & predictions[index - 1]
            if index >= 2:
                recent = predictions[index - 2:index + 1]
                windows["consecutive_3"] = recent[0] & recent[1] & recent[2]
                windows["majority_3"] = np.stack(recent).sum(axis=0) >= 2
            for gate, selected in windows.items():
                tp = int((selected & final_label).sum())
                fp = int((selected & ~final_label).sum())
                positives = int(final_label.sum())
                negatives = int((~final_label).sum())
                schedule_rows.append({
                    "candidate_id": candidate_id, "scope": scope_name,
                    "call_index": index, "t": history[index]["t"],
                    "time_bin": time_bin(history[index]["t"]), "gate": gate,
                    "action_rate": float(selected.mean()),
                    "precision": ratio(tp, tp + fp), "recall": ratio(tp, positives),
                    "false_activation_rate": ratio(fp, negatives),
                })
    schedules = pd.DataFrame(schedule_rows)
    schedules.to_csv(args.output / "persistence_gate_metrics.csv", index=False)
    schedule_summary = schedules.groupby(["scope", "time_bin", "gate"], sort=False).agg(
        candidates=("candidate_id", "nunique"), action_rate=("action_rate", "mean"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        false_activation_rate=("false_activation_rate", "mean"),
    ).reset_index()
    schedule_summary.to_csv(args.output / "persistence_gate_summary.csv", index=False)
    lines = ["# Subgroup trajectory readability", "",
             "Labels are final generated-row membership, not semantic ground truth. The probe uses unconditional sampling and one seed.", "",
             "| Scope | Time bin | Candidates | Evaluable | Accuracy | Precision | Recall | Final positive rate |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.scope} | {row.time_bin} | {row.candidates} | {row.evaluable_candidates} | {row.accuracy:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.positive_rate:.4f} |")
    lines += ["", "## Persistence-gate ablation", "",
              "The gate labels synthetic rows only; it does not yet measure downstream actuation.", "",
              "| Scope | Time bin | Gate | Precision | Recall | False activation | Action rate |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in schedule_summary.itertuples(index=False):
        lines.append(f"| {row.scope} | {row.time_bin} | {row.gate} | {row.precision:.4f} | {row.recall:.4f} | {row.false_activation_rate:.4f} | {row.action_rate:.4f} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "status": "executed", "dataset": args.dataset, "samples": args.samples,
        "steps": args.steps, "seed": args.seed, "candidate_ids": candidates.candidate_id.tolist(),
        "interpretation": "temporal consistency and persistence gating to final generated row; no LLM and no real-row labels",
    }, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
