#!/usr/bin/env python3
"""Probe temporal persistence and selective late rechecks in TabDiff.

This follow-up deliberately stays rule/oracle based.  It asks whether the
mechanism needed by an eventual LLM critic is sound before paying for one:

1. Does a violation that persists across adjacent previews better predict the
   final violation than a one-frame observation?
2. Does a second, targeted late recheck prevent denoising from undoing an
   earlier typed correction at lower critic cost than checking every row twice?

The variants are mechanism proxies, not reproductions of published methods.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import tabdiff_probe as core


RULES = ("education_fd", "husband_male", "wife_female", "spouse_marital")
RELATIONSHIP_RULES = {"husband_male", "wife_female", "spouse_marital"}


def bool_masks_from_preview(
    bundle: core.ModelBundle,
    context: core.RuleContext,
    denoised: torch.Tensor,
    cat_pred: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    preview = core.decode_tensor(bundle, torch.cat([denoised.detach(), cat_pred.detach().float()], dim=1))
    flags = core.rule_flags(preview, context)
    masks = {
        rule: torch.as_tensor(flags[rule].to_numpy(dtype=bool), device=bundle.device)
        for rule in RULES
    }
    parents = {
        "education": cat_pred[:, bundle.cat_name_to_idx["education"]].detach().clone(),
        "relationship": cat_pred[:, bundle.cat_name_to_idx["relationship"]].detach().clone(),
    }
    return masks, parents


def apply_rule_masks(
    bundle: core.ModelBundle,
    denoised: torch.Tensor,
    cat_pred: torch.Tensor,
    x_cat_next: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    corrected_num = denoised.clone()
    corrected_cat = x_cat_next.clone()

    edu_cat_idx = bundle.cat_name_to_idx["education"]
    edu_num_idx = bundle.num_name_to_idx["education.num"]
    edu_codes = cat_pred[:, edu_cat_idx].long().clamp(0, len(bundle.education_target_by_code) - 1)
    targets = bundle.education_target_by_code[edu_codes]
    edu_rows = masks["education_fd"] & torch.isfinite(targets)
    corrected_num[edu_rows, edu_num_idx] = targets[edu_rows]

    sex_idx = bundle.cat_name_to_idx["sex"]
    marital_idx = bundle.cat_name_to_idx["marital.status"]
    male = bundle.category_codes["sex"]["Male"]
    female = bundle.category_codes["sex"]["Female"]
    spouse = bundle.category_codes["marital.status"]["Married-civ-spouse"]
    corrected_cat[masks["husband_male"], sex_idx] = male
    corrected_cat[masks["wife_female"], sex_idx] = female
    corrected_cat[masks["spouse_marital"], marital_idx] = spouse
    events = int(sum(mask.sum().item() for mask in masks.values()))
    return corrected_num, corrected_cat, events


def empty_masks(n: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {rule: torch.zeros(n, dtype=torch.bool, device=device) for rule in RULES}


def stable_persistent_masks(
    current: dict[str, torch.Tensor],
    previous: dict[str, torch.Tensor],
    parents: dict[str, torch.Tensor],
    previous_parents: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    edu_stable = parents["education"].eq(previous_parents["education"])
    rel_stable = parents["relationship"].eq(previous_parents["relationship"])
    result = {}
    for rule in RULES:
        stable = rel_stable if rule in RELATIONSHIP_RULES else edu_stable
        result[rule] = current[rule] & previous[rule] & stable
    return result


def sample_selective(
    bundle: core.ModelBundle,
    context: core.RuleContext,
    n: int,
    steps: int,
    seed: int,
    mode: str,
    intervention_t: float,
    recheck_t: float,
    collect_forecast: bool,
    numeric_actuator: str = "x0_prediction",
):
    diffusion = bundle.diffusion
    device = bundle.device
    core.seed_everything(seed)
    t, sigma_num, sigma_cat, sigma_num_next, sigma_cat_next = core.sampler_grid(diffusion, steps, device)
    x_num = torch.randn((n, diffusion.num_numerical_features), device=device) * sigma_num[-1]
    x_cat = diffusion._sample_masked_prior(n, len(diffusion.num_classes))
    intervention_i = int(torch.argmin((t.squeeze() - intervention_t).abs()).item())
    recheck_i = int(torch.argmin((t.squeeze() - recheck_t).abs()).item())

    previous_masks = None
    previous_parents = None
    initial_masks = empty_masks(n, device)
    forecast_trace = []
    correction_events = 0
    critic_record_evaluations = 0
    rule_record_evaluations = 0
    main_flagged_events = 0
    recheck_flagged_events = 0
    core.synchronize(device)
    started = time.perf_counter()

    for i in reversed(range(steps)):
        with torch.no_grad():
            x_input, denoised, raw_logits, cat_pred = core.model_preview(
                bundle, x_num, x_cat, t[i], sigma_num[i]
            )
            current_masks, parents = bool_masks_from_preview(bundle, context, denoised, cat_pred)
            if collect_forecast:
                forecast_trace.append(
                    {
                        "t": float(t[i].item()),
                        "masks": {k: v.detach().cpu().numpy() for k, v in current_masks.items()},
                        "parents": {k: v.detach().cpu().numpy() for k, v in parents.items()},
                    }
                )

            d_cur = (x_input - denoised) / sigma_num[i]
            x_num_next = x_input + (sigma_num_next[i] - sigma_num[i]) * d_cur
            logits = diffusion._subs_parameterization(raw_logits, x_cat)
            alpha_t = torch.exp(-sigma_cat[i]).unsqueeze(0).repeat(n, 1)
            alpha_s = torch.exp(-sigma_cat_next[i]).unsqueeze(0).repeat(n, 1)
            x_cat_next, _ = diffusion._mdlm_update(logits, x_cat, alpha_t, alpha_s)

            selected = None
            if mode == "fixed_rule_each_step" and float(t[i]) <= 0.75:
                selected = current_masks
                rule_record_evaluations += n
            elif i == intervention_i and mode in {
                "late_once",
                "persistent_late_once",
                "late_recheck_flagged",
                "late_recheck_all",
            }:
                critic_record_evaluations += n
                rule_record_evaluations += n
                if mode == "persistent_late_once":
                    # A temporal-consensus critic must observe both frames.
                    critic_record_evaluations += n
                    if previous_masks is None or previous_parents is None:
                        selected = empty_masks(n, device)
                    else:
                        selected = stable_persistent_masks(
                            current_masks, previous_masks, parents, previous_parents
                        )
                else:
                    selected = current_masks
                initial_masks = {rule: mask.detach().clone() for rule, mask in selected.items()}
                main_flagged_events = int(sum(mask.sum().item() for mask in selected.values()))
            elif i == recheck_i and mode in {"late_recheck_flagged", "late_recheck_all"}:
                if mode == "late_recheck_flagged":
                    eligible_rows = torch.stack(list(initial_masks.values())).any(dim=0)
                    critic_record_evaluations += int(eligible_rows.sum().item())
                    selected = {
                        rule: current_masks[rule] & initial_masks[rule]
                        for rule in RULES
                    }
                else:
                    critic_record_evaluations += n
                    selected = current_masks
                rule_record_evaluations += n
                recheck_flagged_events = int(sum(mask.sum().item() for mask in selected.values()))

            if selected is not None:
                corrected_num, corrected_cat, changed = apply_rule_masks(
                    bundle, denoised, cat_pred, x_cat_next, selected
                )
                correction_events += changed
                if numeric_actuator == "x0_prediction":
                    d_cur = (x_input - corrected_num) / sigma_num[i]
                    x_num_next = x_input + (sigma_num_next[i] - sigma_num[i]) * d_cur
                elif numeric_actuator == "state_projection":
                    edu_cat_idx = bundle.cat_name_to_idx["education"]
                    edu_num_idx = bundle.num_name_to_idx["education.num"]
                    edu_codes = cat_pred[:, edu_cat_idx].long().clamp(
                        0, len(bundle.education_target_by_code) - 1
                    )
                    targets = bundle.education_target_by_code[edu_codes]
                    rows = selected["education_fd"] & torch.isfinite(targets)
                    x_num_next[rows, edu_num_idx] = targets[rows]
                else:
                    raise ValueError(f"Unknown numeric_actuator={numeric_actuator}")
                x_cat_next = corrected_cat

            previous_masks = {rule: mask.detach().clone() for rule, mask in current_masks.items()}
            previous_parents = {key: value.detach().clone() for key, value in parents.items()}
            x_num = x_num_next.detach()
            x_cat = x_cat_next.detach()

    core.synchronize(device)
    elapsed = time.perf_counter() - started
    final_encoded = torch.cat([x_num, x_cat.float()], dim=1).cpu()
    final_df = core.decode_tensor(bundle, final_encoded)
    final_flags_df = core.rule_flags(final_df, context)
    final_masks = {
        rule: torch.as_tensor(final_flags_df[rule].to_numpy(dtype=bool), device=device)
        for rule in RULES
    }
    initial_total = int(sum(mask.sum().item() for mask in initial_masks.values()))
    residual = int(sum((initial_masks[r] & final_masks[r]).sum().item() for r in RULES))
    diagnostics = {
        "critic_record_evaluations_proxy": critic_record_evaluations,
        "rule_record_evaluations": rule_record_evaluations,
        "correction_relation_events": correction_events,
        "main_flagged_relation_events": main_flagged_events,
        "recheck_flagged_relation_events": recheck_flagged_events,
        "residual_among_initial_actions": residual / initial_total if initial_total else math.nan,
    }
    for rule in RULES:
        rule_initial = int(initial_masks[rule].sum().item())
        rule_residual = int((initial_masks[rule] & final_masks[rule]).sum().item())
        diagnostics[f"initial_{rule}_events"] = rule_initial
        diagnostics[f"residual_{rule}_events"] = rule_residual
        diagnostics[f"residual_{rule}_rate"] = (
            rule_residual / rule_initial if rule_initial else math.nan
        )
        diagnostics[f"new_final_{rule}_events"] = int(
            ((~initial_masks[rule]) & final_masks[rule]).sum().item()
        )
    return final_df, final_encoded, elapsed, diagnostics, forecast_trace


def binary_scores(pred: np.ndarray, gold: np.ndarray) -> dict[str, float | int]:
    pred = pred.astype(bool)
    gold = gold.astype(bool)
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    tn = int((~pred & ~gold).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": float(pred.mean()),
        "final_positive_rate": float(gold.mean()),
    }


def forecast_metrics(
    trace: list[dict],
    final_df: pd.DataFrame,
    context: core.RuleContext,
    seed: int,
) -> list[dict]:
    final_flags = core.rule_flags(final_df, context)
    rows = []
    for position, current in enumerate(trace):
        previous = trace[position - 1] if position > 0 else None
        following = trace[position + 1] if position + 1 < len(trace) else None
        for rule in RULES:
            gold = final_flags[rule].to_numpy(dtype=bool)
            raw = current["masks"][rule].astype(bool)
            raw_scores = binary_scores(raw, gold)
            rows.append(
                {
                    "seed": seed,
                    "t": current["t"],
                    "rule": rule,
                    "gate": "instantaneous",
                    "parent_stable_rate": math.nan,
                    "flip_to_next_rate": float((raw != following["masks"][rule]).mean()) if following else math.nan,
                    **raw_scores,
                }
            )
            if previous is not None:
                parent = "relationship" if rule in RELATIONSHIP_RULES else "education"
                stable = current["parents"][parent] == previous["parents"][parent]
                persistent = raw & previous["masks"][rule].astype(bool) & stable
                rows.append(
                    {
                        "seed": seed,
                        "t": current["t"],
                        "rule": rule,
                        "gate": "persistent_and_parent_stable",
                        "parent_stable_rate": float(stable.mean()),
                        "flip_to_next_rate": float((raw != following["masks"][rule]).mean()) if following else math.nan,
                        **binary_scores(persistent, gold),
                    }
                )
    return rows


def plot_forecast(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, rule in zip(axes.flat, RULES):
        subset = frame[frame["rule"].eq(rule)]
        for gate, style in (("instantaneous", "-"), ("persistent_and_parent_stable", "--")):
            selected = subset[subset["gate"].eq(gate)]
            grouped = selected.groupby("t", as_index=False)[["precision", "recall"]].mean().sort_values("t")
            ax.plot(grouped["t"], grouped["precision"], style, label=f"{gate}: precision")
            ax.plot(grouped["t"], grouped["recall"], style, alpha=0.55, label=f"{gate}: recall")
        ax.set_title(rule)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("reverse trajectory t")
        ax.set_ylabel("forecast metric")
    axes.flat[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260922, 20260923, 20260924])
    parser.add_argument("--intervention-t", type=float, default=0.10)
    parser.add_argument("--recheck-t", type=float, default=0.03)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    bundle = core.build_model(args.repo, args.checkpoint, args.config, args.steps, device)
    real = pd.read_csv(args.repo / "synthetic/adult/real.csv")
    test = pd.read_csv(args.repo / "synthetic/adult/test.csv")
    context = core.make_rule_context(real)

    modes = (
        "no_guidance",
        "fixed_rule_each_step",
        "late_once",
        "persistent_late_once",
        "late_recheck_flagged",
        "late_recheck_all",
    )
    metric_rows = []
    forecast_rows = []
    for seed_index, seed in enumerate(args.seeds):
        samples_by_mode = {}
        for mode in modes:
            syn, _, runtime, diagnostics, trace = sample_selective(
                bundle,
                context,
                args.samples,
                args.steps,
                seed,
                mode,
                args.intervention_t,
                args.recheck_t,
                collect_forecast=mode == "no_guidance",
            )
            row = core.evaluate_variant(
                mode,
                syn,
                real,
                test,
                context,
                bundle,
                runtime,
                args.steps,
                diagnostics["correction_relation_events"],
            )
            row.update({"seed": seed, **diagnostics})
            metric_rows.append(row)
            samples_by_mode[mode] = syn
            if mode == "no_guidance":
                forecast_rows.extend(forecast_metrics(trace, syn, context, seed))

        repaired = core.post_repair(samples_by_mode["no_guidance"], context)
        repair_row = core.evaluate_variant(
            "post_repair",
            repaired,
            real,
            test,
            context,
            bundle,
            0.0,
            args.steps,
            int(core.rule_flags(samples_by_mode["no_guidance"], context)["any_hard"].sum()),
        )
        repair_row.update(
            {
                "seed": seed,
                "critic_record_evaluations_proxy": 0,
                "rule_record_evaluations": args.samples,
                "correction_relation_events": math.nan,
                "main_flagged_relation_events": math.nan,
                "recheck_flagged_relation_events": math.nan,
                "residual_among_initial_actions": 0.0,
            }
        )
        metric_rows.append(repair_row)
        if seed_index == 0:
            for mode, frame in {**samples_by_mode, "post_repair": repaired}.items():
                frame.to_csv(args.output / f"samples_{mode}_seed{seed}.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    forecasts = pd.DataFrame(forecast_rows)
    metrics.to_csv(args.output / "intervention_metrics_by_seed.csv", index=False)
    forecasts.to_csv(args.output / "trajectory_violation_forecast.csv", index=False)
    numeric = [c for c in metrics.select_dtypes(include=[np.number]).columns if c != "seed"]
    summary = metrics.groupby("variant")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.output / "intervention_metrics_summary.csv", index=False)
    plot_forecast(forecasts, args.output / "trajectory_violation_forecast.png")

    metadata = {
        "status": "executed",
        "scope": "rule/oracle mechanism probe; no LLM calls; not a paper reproduction",
        "checkpoint": str(args.checkpoint),
        "checkpoint_training_steps": pickle.load(args.config.open("rb"))["train"]["main"]["steps"],
        "seeds": args.seeds,
        "samples_per_seed": args.samples,
        "steps": args.steps,
        "intervention_t": args.intervention_t,
        "recheck_t": args.recheck_t,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    display = [
        "variant",
        "hard_any_violation_rate_mean",
        "numeric_ks_mean_mean",
        "categorical_tvd_mean_mean",
        "rare_pair_coverage_mean",
        "critic_record_evaluations_proxy_mean",
        "residual_among_initial_actions_mean",
    ]
    print(summary[display].to_string(index=False))


if __name__ == "__main__":
    main()
