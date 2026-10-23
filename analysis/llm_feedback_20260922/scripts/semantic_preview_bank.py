#!/usr/bin/env python3
"""Build a synthetic-only paired semantic stress-test bank along trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import tabdiff_probe as core


TARGET_TIMES = (0.40, 0.25, 0.17, 0.10, 0.03)
CORRUPTIONS = (
    "education_fd",
    "numeric_domain",
    "spouse_consistency",
    "age_education_plausibility",
    "missingness_coupling",
)


def python_record(row: pd.Series) -> dict:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in row.to_dict().items()
    }


def semantic_relations(record: dict, education_map: dict[str, float]) -> set[str]:
    clean = {
        key: value.strip() if isinstance(value, str) else value
        for key, value in record.items()
    }
    found = set()
    education = str(clean.get("education", ""))
    education_num = pd.to_numeric(pd.Series([clean.get("education.num")]), errors="coerce").iloc[0]
    expected = education_map.get(education)
    if expected is not None and (pd.isna(education_num) or not np.isclose(float(education_num), expected)):
        found.add("education_fd")
    for column in ("capital.gain", "capital.loss"):
        value = pd.to_numeric(pd.Series([clean.get(column)]), errors="coerce").iloc[0]
        if pd.isna(value) or value < 0:
            found.add("numeric_domain")
    hours = pd.to_numeric(pd.Series([clean.get("hours.per.week")]), errors="coerce").iloc[0]
    if pd.isna(hours) or hours < 1:
        found.add("numeric_domain")
    relation = clean.get("relationship")
    if relation == "Husband" and (
        clean.get("sex") != "Male" or clean.get("marital.status") != "Married-civ-spouse"
    ):
        found.add("spouse_consistency")
    if relation == "Wife" and (
        clean.get("sex") != "Female" or clean.get("marital.status") != "Married-civ-spouse"
    ):
        found.add("spouse_consistency")
    age = pd.to_numeric(pd.Series([clean.get("age")]), errors="coerce").iloc[0]
    if education in {"Doctorate", "Prof-school"} and (pd.isna(age) or age < 20):
        found.add("age_education_plausibility")
    if education == "Masters" and (pd.isna(age) or age < 18):
        found.add("age_education_plausibility")
    work_missing = str(clean.get("workclass", "")) == "?"
    occupation_missing = str(clean.get("occupation", "")) == "?"
    if work_missing != occupation_missing:
        found.add("missingness_coupling")
    return found


def corrupt(record: dict, corruption_type: str, education_map: dict[str, float]) -> dict:
    result = dict(record)
    if corruption_type == "education_fd":
        education = str(result["education"]).strip()
        expected = education_map.get(education, 9.0)
        result["education.num"] = float(expected + (1 if expected < 16 else -1))
    elif corruption_type == "numeric_domain":
        result["capital.loss"] = -500.0
    elif corruption_type == "spouse_consistency":
        result["relationship"] = " Husband"
        result["sex"] = " Female"
        result["marital.status"] = " Never-married"
    elif corruption_type == "age_education_plausibility":
        result["age"] = 14.0
        result["education"] = " Doctorate"
        result["education.num"] = 16.0
    elif corruption_type == "missingness_coupling":
        result["workclass"] = " ?"
        result["occupation"] = " Exec-managerial"
    else:
        raise ValueError(corruption_type)
    return result


def changed_columns(previous: dict | None, current: dict) -> list[str]:
    if previous is None:
        return []
    changed = []
    for key, value in current.items():
        other = previous.get(key)
        if isinstance(value, (float, int)) and isinstance(other, (float, int)):
            if not np.isclose(float(value), float(other), rtol=1e-4, atol=1e-3):
                changed.append(key)
        elif str(value).strip() != str(other).strip():
            changed.append(key)
    return changed


def make_cases(
    frame: pd.DataFrame,
    previous: pd.DataFrame | None,
    t_value: float,
    seed: int,
    education_map: dict[str, float],
    cases_per_type: int,
) -> list[dict]:
    records = [python_record(frame.iloc[index]) for index in range(len(frame))]
    previous_records = (
        [python_record(previous.iloc[index]) for index in range(len(previous))]
        if previous is not None
        else [None] * len(frame)
    )
    valid_indices = [
        index for index, record in enumerate(records)
        if not semantic_relations(record, education_map)
    ]
    if len(valid_indices) < len(CORRUPTIONS) * cases_per_type + 4:
        raise RuntimeError(f"Only {len(valid_indices)} clean preview rows at t={t_value}")

    cases = []
    cursor = 0
    for corruption_type in CORRUPTIONS:
        for local_index in range(cases_per_type):
            row_index = valid_indices[cursor]
            cursor += 1
            clean_record = records[row_index]
            prior_record = previous_records[row_index]
            corrupted = corrupt(clean_record, corruption_type, education_map)
            gold = sorted(semantic_relations(corrupted, education_map))
            semantic_key = f"seed{seed}_row{row_index}_{corruption_type}"
            pair_id = f"{semantic_key}_t{t_value:.3f}"
            common = {
                "trajectory_id": f"seed{seed}_row{row_index}",
                "semantic_key": semantic_key,
                "pair_id": pair_id,
                "t": t_value,
                "previous_record": prior_record,
                "intended_relation": corruption_type,
                "privacy": "synthetic_preview_only",
            }
            cases.append(
                {
                    **common,
                    "case_id": f"{pair_id}_corrupt",
                    "control_type": "corrupted",
                    "record": corrupted,
                    "gold_relations": gold,
                    "changed_columns_from_previous": changed_columns(prior_record, corrupted),
                }
            )
            clean_gold = sorted(semantic_relations(clean_record, education_map))
            cases.append(
                {
                    **common,
                    "case_id": f"{pair_id}_clean",
                    "control_type": "paired_clean",
                    "record": clean_record,
                    "gold_relations": clean_gold,
                    "changed_columns_from_previous": changed_columns(prior_record, clean_record),
                }
            )

    # Rare-but-valid counterexamples guard against typicality-driven repair.
    for control_type in ("rare_valid_country", "exception_valid_own_child"):
        for local_index in range(2):
            row_index = valid_indices[cursor]
            cursor += 1
            record = dict(records[row_index])
            if control_type == "rare_valid_country":
                record["native.country"] = " France"
            else:
                record["relationship"] = " Own-child"
                record["age"] = 45.0 + local_index
                record["marital.status"] = " Never-married"
            prior_record = previous_records[row_index]
            case_id = f"seed{seed}_row{row_index}_{control_type}_t{t_value:.3f}"
            cases.append(
                {
                    "case_id": case_id,
                    "trajectory_id": f"seed{seed}_row{row_index}",
                    "semantic_key": f"seed{seed}_row{row_index}_{control_type}",
                    "pair_id": case_id,
                    "t": t_value,
                    "previous_record": prior_record,
                    "intended_relation": "none",
                    "privacy": "synthetic_preview_only",
                    "control_type": control_type,
                    "record": record,
                    "gold_relations": sorted(semantic_relations(record, education_map)),
                    "changed_columns_from_previous": changed_columns(prior_record, record),
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260931)
    parser.add_argument("--cases-per-type", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    bundle = core.build_model(args.repo, args.checkpoint, args.config, args.steps, device)
    real = pd.read_csv(args.repo / "synthetic/adult/real.csv")
    context = core.make_rule_context(real)
    diffusion = bundle.diffusion
    core.seed_everything(args.seed)
    t, sigma_num, sigma_cat, sigma_num_next, sigma_cat_next = core.sampler_grid(
        diffusion, args.steps, device
    )
    x_num = torch.randn((args.samples, diffusion.num_numerical_features), device=device) * sigma_num[-1]
    x_cat = diffusion._sample_masked_prior(args.samples, len(diffusion.num_classes))
    target_indices = {
        int(torch.argmin((t.squeeze() - target).abs()).item()): target
        for target in TARGET_TIMES
    }
    previous_preview = None
    cases = []
    snapshots = []
    for index in reversed(range(args.steps)):
        with torch.no_grad():
            x_input, denoised, raw_logits, cat_pred = core.model_preview(
                bundle, x_num, x_cat, t[index], sigma_num[index]
            )
            preview = core.decode_tensor(
                bundle, torch.cat([denoised.detach(), cat_pred.detach().float()], dim=1)
            )
            if index in target_indices:
                actual_t = float(t[index].item())
                cases.extend(
                    make_cases(
                        preview,
                        previous_preview,
                        actual_t,
                        args.seed,
                        context.education_map,
                        args.cases_per_type,
                    )
                )
                snapshots.append({"requested_t": target_indices[index], "actual_t": actual_t})

            d_cur = (x_input - denoised) / sigma_num[index]
            x_num = (x_input + (sigma_num_next[index] - sigma_num[index]) * d_cur).detach()
            logits = diffusion._subs_parameterization(raw_logits, x_cat)
            alpha_t = torch.exp(-sigma_cat[index]).unsqueeze(0).repeat(args.samples, 1)
            alpha_s = torch.exp(-sigma_cat_next[index]).unsqueeze(0).repeat(args.samples, 1)
            x_cat, _ = diffusion._mdlm_update(logits, x_cat, alpha_t, alpha_s)
            x_cat = x_cat.detach()
            previous_preview = preview

    with (args.output / "semantic_cases.jsonl").open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")
    summary = pd.DataFrame(
        [
            {
                "t": case["t"],
                "control_type": case["control_type"],
                "intended_relation": case["intended_relation"],
                "has_gold_violation": bool(case["gold_relations"]),
            }
            for case in cases
        ]
    ).groupby(["t", "control_type", "intended_relation", "has_gold_violation"]).size().rename("count").reset_index()
    summary.to_csv(args.output / "bank_summary.csv", index=False)
    metadata = {
        "status": "executed",
        "scope": "synthetic previews and synthetic counterfactuals only; no real rows",
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "samples": args.samples,
        "cases": len(cases),
        "snapshots": snapshots,
        "corruptions": list(CORRUPTIONS),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
