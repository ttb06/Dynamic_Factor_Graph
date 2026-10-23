#!/usr/bin/env python3
"""Controlled end-to-end LLM-gated rollback stress test inside TabDiff sampling."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import tabdiff_probe as core
from llm_feedback_audit import load_env
from semantic_llm_sweep import call_batch
from semantic_preview_bank import semantic_relations


CORRUPTION_TYPES = (
    "education_fd",
    "spouse_consistency",
    "age_education_plausibility",
    "missingness_coupling",
)
RELATION_COLUMNS = {
    "education_fd": ["education", "education.num"],
    "spouse_consistency": ["relationship", "sex", "marital.status"],
    "age_education_plausibility": ["age", "education", "education.num"],
    "missingness_coupling": ["workclass", "occupation"],
}


def mutate_state(bundle: core.ModelBundle, x_num: torch.Tensor, x_cat: torch.Tensor):
    n = len(x_num)
    if n % len(CORRUPTION_TYPES):
        raise ValueError("sample count must be divisible by corruption type count")
    per_type = n // len(CORRUPTION_TYPES)
    labels = []
    original_num = x_num.detach().clone()
    original_cat = x_cat.detach().clone()
    raw_num = bundle.dataset_train.num_inverse(x_num.detach().cpu().numpy()).astype(float)

    age_idx = bundle.num_name_to_idx["age"]
    edu_num_idx = bundle.num_name_to_idx["education.num"]
    edu_idx = bundle.cat_name_to_idx["education"]
    relationship_idx = bundle.cat_name_to_idx["relationship"]
    sex_idx = bundle.cat_name_to_idx["sex"]
    marital_idx = bundle.cat_name_to_idx["marital.status"]
    workclass_idx = bundle.cat_name_to_idx["workclass"]
    occupation_idx = bundle.cat_name_to_idx["occupation"]

    for group, corruption in enumerate(CORRUPTION_TYPES):
        start, stop = group * per_type, (group + 1) * per_type
        rows = slice(start, stop)
        labels.extend([corruption] * per_type)
        if corruption == "education_fd":
            x_cat[rows, edu_idx] = bundle.category_codes["education"]["HS-grad"]
            raw_num[rows, edu_num_idx] = 16.0
        elif corruption == "spouse_consistency":
            x_cat[rows, relationship_idx] = bundle.category_codes["relationship"]["Husband"]
            x_cat[rows, sex_idx] = bundle.category_codes["sex"]["Female"]
            x_cat[rows, marital_idx] = bundle.category_codes["marital.status"]["Never-married"]
        elif corruption == "age_education_plausibility":
            raw_num[rows, age_idx] = 14.0
            raw_num[rows, edu_num_idx] = 16.0
            x_cat[rows, edu_idx] = bundle.category_codes["education"]["Doctorate"]
        elif corruption == "missingness_coupling":
            x_cat[rows, workclass_idx] = bundle.category_codes["workclass"]["?"]
            x_cat[rows, occupation_idx] = bundle.category_codes["occupation"]["Exec-managerial"]
    transformed = bundle.dataset_train.num_transform.transform(raw_num)
    x_num = torch.as_tensor(transformed, dtype=x_num.dtype, device=x_num.device)
    return x_num, x_cat, original_num, original_cat, labels


def restore_relation_columns(
    bundle: core.ModelBundle,
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    original_num: torch.Tensor,
    original_cat: torch.Tensor,
    row: int,
    relation: str,
) -> None:
    for column in RELATION_COLUMNS.get(relation, []):
        if column in bundle.num_name_to_idx:
            index = bundle.num_name_to_idx[column]
            x_num[row, index] = original_num[row, index]
        elif column in bundle.cat_name_to_idx:
            index = bundle.cat_name_to_idx[column]
            x_cat[row, index] = original_cat[row, index]


def decide_with_llm(
    key: str,
    model: str,
    condition: str,
    cases: list[dict],
    batch_size: int,
    delay: float,
):
    decisions = {}
    raw = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    latency_total = 0.0
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        parsed, usage, latency, error = {}, {}, math.nan, None
        for attempt in range(5):
            try:
                parsed, usage, latency = call_batch(key, model, batch, condition)
                error = None
                break
            except urllib.error.HTTPError as exc:
                error = f"HTTPError {exc.code}: {exc.read().decode('utf-8', errors='replace')[:300]}"
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
                retry_after = float(exc.headers.get("Retry-After", 0) or 0)
                time.sleep(min(30.0, max(retry_after, 2**attempt + random.random())))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(30.0, 2**attempt + random.random()))
        if error is None:
            latency_total += latency
            for name in usage_total:
                usage_total[name] += int(usage.get(name, 0) or 0)
            for item in parsed.get("decisions", []):
                decisions[item.get("case_id")] = item
        raw.append(
            {
                "batch_start": start,
                "case_ids": [case["case_id"] for case in batch],
                "parsed": parsed,
                "usage": usage,
                "latency": latency,
                "error": error,
            }
        )
        if start + batch_size < len(cases):
            time.sleep(delay)
    return decisions, raw, usage_total, latency_total


def sample_branch(
    bundle: core.ModelBundle,
    context: core.RuleContext,
    n: int,
    steps: int,
    seed: int,
    target_t: float,
    branch: str,
    api: dict | None,
):
    diffusion = bundle.diffusion
    device = bundle.device
    core.seed_everything(seed)
    t, sigma_num, sigma_cat, sigma_num_next, sigma_cat_next = core.sampler_grid(diffusion, steps, device)
    target_index = int(torch.argmin((t.squeeze() - target_t).abs()).item())
    x_num = torch.randn((n, diffusion.num_numerical_features), device=device) * sigma_num[-1]
    x_cat = diffusion._sample_masked_prior(n, len(diffusion.num_classes))
    previous_preview = None
    labels = ["none"] * n
    llm_details = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    api_latency = 0.0

    for index in reversed(range(steps)):
        if index == target_index and branch != "clean_no_injection":
            x_num, x_cat, original_num, original_cat, labels = mutate_state(
                bundle, x_num.clone(), x_cat.clone()
            )
            with torch.no_grad():
                _, preview_num, _, preview_cat = core.model_preview(
                    bundle, x_num, x_cat, t[index], sigma_num[index]
                )
                preview = core.decode_tensor(
                    bundle, torch.cat([preview_num, preview_cat.float()], dim=1)
                )
            if branch in {"oracle_rollback", "static_verifier_rollback"}:
                for row, relation in enumerate(labels):
                    if branch == "oracle_rollback" or relation in semantic_relations(
                        preview.iloc[row].to_dict(), context.education_map
                    ):
                        restore_relation_columns(
                            bundle, x_num, x_cat, original_num, original_cat, row, relation
                        )
            elif branch == "llm_rollback":
                cases = []
                for row in range(n):
                    record = {
                        key: value.item() if isinstance(value, np.generic) else value
                        for key, value in preview.iloc[row].to_dict().items()
                    }
                    prior = None
                    if previous_preview is not None:
                        prior = {
                            key: value.item() if isinstance(value, np.generic) else value
                            for key, value in previous_preview.iloc[row].to_dict().items()
                        }
                    cases.append(
                        {
                            "case_id": f"row{row}",
                            "t": float(t[index]),
                            "record": record,
                            "previous_record": prior,
                            "changed_columns_from_previous": [],
                        }
                    )
                decisions, llm_details, usage, api_latency = decide_with_llm(
                    api["key"], api["model"], api["condition"], cases, api["batch_size"], api["delay"]
                )
                for row, true_relation in enumerate(labels):
                    decision = decisions.get(f"row{row}", {})
                    predicted = decision.get("relation") if decision.get("status") == "violated" else None
                    if predicted in RELATION_COLUMNS:
                        restore_relation_columns(
                            bundle, x_num, x_cat, original_num, original_cat, row, predicted
                        )
                    llm_details.append(
                        {
                            "row": row,
                            "true_injected_relation": true_relation,
                            "predicted_relation": predicted,
                            "status": decision.get("status", "missing"),
                            "confidence": decision.get("confidence"),
                        }
                    )

        with torch.no_grad():
            x_input, denoised, raw_logits, cat_pred = core.model_preview(
                bundle, x_num, x_cat, t[index], sigma_num[index]
            )
            preview_now = core.decode_tensor(
                bundle, torch.cat([denoised, cat_pred.float()], dim=1)
            )
            d_cur = (x_input - denoised) / sigma_num[index]
            x_num = (x_input + (sigma_num_next[index] - sigma_num[index]) * d_cur).detach()
            logits = diffusion._subs_parameterization(raw_logits, x_cat)
            alpha_t = torch.exp(-sigma_cat[index]).unsqueeze(0).repeat(n, 1)
            alpha_s = torch.exp(-sigma_cat_next[index]).unsqueeze(0).repeat(n, 1)
            x_cat, _ = diffusion._mdlm_update(logits, x_cat, alpha_t, alpha_s)
            x_cat = x_cat.detach()
            previous_preview = preview_now

    final_encoded = torch.cat([x_num, x_cat.float()], dim=1).cpu()
    final_df = core.decode_tensor(bundle, final_encoded)
    return final_df, final_encoded, labels, llm_details, usage, api_latency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--target-t", type=float, required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260941)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--inter-call-delay", type=float, default=4.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")
    device = torch.device("cuda")
    bundle = core.build_model(args.repo, args.checkpoint, args.config, args.steps, device)
    real = pd.read_csv(args.repo / "synthetic/adult/real.csv")
    test = pd.read_csv(args.repo / "synthetic/adult/test.csv")
    context = core.make_rule_context(real)
    api = {
        "key": key,
        "model": args.model,
        "condition": args.condition,
        "batch_size": args.batch_size,
        "delay": args.inter_call_delay,
    }

    branches = [
        "clean_no_injection",
        "corrupt_no_feedback",
        "oracle_rollback",
        "static_verifier_rollback",
        "llm_rollback",
    ]
    outputs = {}
    branch_meta = {}
    for branch in branches:
        frame, encoded, labels, details, usage, latency = sample_branch(
            bundle,
            context,
            args.samples,
            args.steps,
            args.seed,
            args.target_t,
            branch,
            api if branch == "llm_rollback" else None,
        )
        outputs[branch] = (frame, encoded, labels)
        branch_meta[branch] = {"usage": usage, "api_latency_seconds": latency}
        frame.to_csv(args.output / f"samples_{branch}.csv", index=False)
        if branch == "llm_rollback":
            with (args.output / "llm_details.jsonl").open("w") as handle:
                for item in details:
                    handle.write(json.dumps(item) + "\n")

    clean_encoded = outputs["clean_no_injection"][1]
    rows = []
    for branch, (frame, encoded, labels) in outputs.items():
        relations = [semantic_relations(frame.iloc[i].to_dict(), context.education_map) for i in range(len(frame))]
        for corruption in CORRUPTION_TYPES:
            indices = [i for i, label in enumerate(labels) if label == corruption]
            if not indices:
                indices = list(range(len(frame))) if branch == "clean_no_injection" else []
            intended_violation = [corruption in relations[i] for i in indices]
            rows.append(
                {
                    "branch": branch,
                    "corruption_type": corruption,
                    "rows": len(indices),
                    "final_intended_violation_rate": float(np.mean(intended_violation)) if indices else math.nan,
                    "final_any_semantic_violation_rate": float(np.mean([bool(relations[i]) for i in indices])) if indices else math.nan,
                    "numeric_mae_to_clean_encoded": float(
                        (encoded[indices, : bundle.dataset_train.d_numerical] - clean_encoded[indices, : bundle.dataset_train.d_numerical]).abs().mean()
                    ) if indices else math.nan,
                    "categorical_agreement_to_clean": float(
                        (encoded[indices, bundle.dataset_train.d_numerical :].long() == clean_encoded[indices, bundle.dataset_train.d_numerical :].long()).float().mean()
                    ) if indices else math.nan,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "rollback_metrics.csv", index=False)
    metadata = {
        "status": "executed",
        "scope": "controlled synthetic latent corruption; pre-corruption column rollback upper bound",
        "condition": args.condition,
        "model": args.model,
        "target_t": args.target_t,
        "samples": args.samples,
        "seed": args.seed,
        "branch_metadata": branch_meta,
        "privacy_boundary": "Only synthetic preview rows were sent to API; no real row.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
