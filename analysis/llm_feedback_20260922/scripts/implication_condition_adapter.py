#!/usr/bin/env python3
"""Test antecedent evasion in conditional semantic-contract diffusion.

Two training constructions are compared:
``filter`` drops violating rows and can satisfy a contract by deleting its
antecedent subgroup; ``repair_preserve`` repairs only the mutable consequent
while preserving antecedent and exception prevalence.
"""

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
import torch.nn as nn
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader, TensorDataset

from conditioned_tabdiff_adapter import (
    ConditionedUniModMLP, ContractConditionedModel, decode, seed_all, total_variation,
)
from implication_contract_catalog import antecedent_mask, catalog, exception_mask, vector, violation_mask
from tabdiff_probe import preprocess_adult


class ApplicabilityGatedConditionedModel(nn.Module):
    """Two-pass row gate: unconditional preview -> applicability -> condition.

    The gate is fully executable for this controlled pilot.  It uses the
    current unmasked category when available and otherwise the unconditional
    x0 categorical prediction.  Thus no LLM is in the sampling loop.
    """

    def __init__(self, base_model, backbone, condition_dim: int, category_sizes, category_codes, category_positions,
                 gate_start_t: float = 1.0, numeric_positions=None):
        super().__init__()
        self.base_model = base_model
        self.backbone = backbone
        self.condition_dim = condition_dim
        self.category_sizes = [int(value) for value in category_sizes]
        self.category_codes = category_codes
        self.category_positions = category_positions
        self.numeric_positions = numeric_positions or {}
        self.condition = None
        self.guidance_scale = 1.0
        self.gate_start_t = float(gate_start_t)
        self.early_policy = "off"
        self.early_floor = 0.0
        self.external_gate = None
        self.external_prior = None
        self.gate_power = 1.0
        self.actuation_scope = "all"
        self.trace_enabled = False
        self.gate_history = []

    def set_condition(self, condition, guidance_scale: float = 1.0):
        self.condition = condition
        self.guidance_scale = float(guidance_scale)

    def start_trace(self):
        self.gate_history = []
        self.trace_enabled = True

    def stop_trace(self):
        self.trace_enabled = False
        return self.gate_history

    def predicted_codes(self, x_cat, raw_logits):
        current_chunks = torch.split(x_cat, [size + 1 for size in self.category_sizes], dim=-1)
        logit_chunks = torch.split(raw_logits, [size + 1 for size in self.category_sizes], dim=-1)
        codes = []
        for size, current, logits in zip(self.category_sizes, current_chunks, logit_chunks):
            current_code = current.argmax(dim=-1)
            predicted = logits[:, :size].argmax(dim=-1)
            codes.append(torch.where(current_code.eq(size), predicted, current_code))
        return torch.stack(codes, dim=1)

    def applicability(self, condition, codes):
        gate = torch.ones(len(condition), dtype=torch.bool, device=condition.device)
        advanced_requested = condition[:, 0].gt(0.5)
        private_requested = condition[:, 1].gt(0.5)
        exception_requested = condition[:, 5].gt(0.5)
        education = codes[:, self.category_positions["education"]]
        workclass = codes[:, self.category_positions["workclass"]]
        occupation = codes[:, self.category_positions["occupation"]]
        advanced_codes = torch.tensor(
            [self.category_codes["education"][value] for value in ("Masters", "Prof-school", "Doctorate")],
            device=codes.device,
        )
        advanced_match = (education[:, None] == advanced_codes[None, :]).any(dim=1)
        private_match = workclass.eq(self.category_codes["workclass"]["Private"])
        exception_match = occupation.eq(self.category_codes["occupation"]["Exec-managerial"])
        gate &= ~advanced_requested | advanced_match
        gate &= ~private_requested | private_match
        gate &= ~exception_requested | ~exception_match
        # Null conditions stay null; this line only makes the intended meaning explicit.
        requested = condition.abs().sum(dim=1).gt(0)
        return gate & requested

    def soft_applicability(self, condition, x_cat, raw_logits):
        """Posterior applicability before categorical argmax becomes stable.

        Observed (unmasked) categories are represented as point masses. Masked
        categories use the denoiser logits. Conjunctive antecedents multiply
        their probabilities; an exception clause multiplies by one minus the
        exception probability.
        """
        current_chunks = torch.split(x_cat, [size + 1 for size in self.category_sizes], dim=-1)
        logit_chunks = torch.split(raw_logits, [size + 1 for size in self.category_sizes], dim=-1)
        probabilities = []
        for size, current, logits in zip(self.category_sizes, current_chunks, logit_chunks):
            current_code = current.argmax(dim=-1)
            posterior = torch.softmax(logits[:, :size], dim=-1)
            observed = current_code.lt(size)
            one_hot = torch.nn.functional.one_hot(current_code.clamp(max=size - 1), num_classes=size).to(posterior.dtype)
            probabilities.append(torch.where(observed[:, None], one_hot, posterior))

        education = probabilities[self.category_positions["education"]]
        workclass = probabilities[self.category_positions["workclass"]]
        occupation = probabilities[self.category_positions["occupation"]]
        advanced_codes = [self.category_codes["education"][value] for value in ("Masters", "Prof-school", "Doctorate")]
        p_advanced = education[:, advanced_codes].sum(dim=1)
        p_private = workclass[:, self.category_codes["workclass"]["Private"]]
        p_exception = occupation[:, self.category_codes["occupation"]["Exec-managerial"]]

        probability = torch.ones(len(condition), device=condition.device, dtype=condition.dtype)
        probability = torch.where(condition[:, 0].gt(0.5), probability * p_advanced, probability)
        probability = torch.where(condition[:, 1].gt(0.5), probability * p_private, probability)
        probability = torch.where(condition[:, 5].gt(0.5), probability * (1 - p_exception), probability)
        requested = condition.abs().sum(dim=1).gt(0).to(probability.dtype)
        return probability * requested

    def forward(self, x_num, x_cat, t, sigma=None):
        condition = self.condition
        if condition is None:
            condition = torch.zeros(len(x_num), self.condition_dim, device=x_num.device, dtype=x_num.dtype)
        self.backbone.set_condition(torch.zeros_like(condition))
        unconditional = self.base_model(x_num, x_cat, t, sigma=sigma)
        codes = self.predicted_codes(x_cat, unconditional[1])
        gate = self.applicability(condition, codes).to(condition.dtype)[:, None]
        soft_gate = self.soft_applicability(condition, x_cat, unconditional[1])[:, None]
        # Reverse sampling runs from t=1 (most noisy) to t=0.  Before the
        # row gate becomes reliable, a schedule probe can either abstain or
        # apply the request globally, then switch to selective applicability.
        if self.early_policy == "external":
            if self.external_gate is None or len(self.external_gate) != len(condition):
                raise ValueError("external gate must match the sampling batch")
            gate = self.external_gate.to(device=condition.device, dtype=condition.dtype).reshape(-1, 1)
        elif self.gate_start_t < 0:
            gate = torch.zeros_like(gate)
        else:
            late = t.reshape(-1).le(self.gate_start_t)[:, None]
            if self.early_policy == "global":
                requested = condition.abs().sum(dim=1).gt(0)[:, None]
                gate = torch.where(late, gate.bool(), requested).to(gate.dtype)
            elif self.early_policy == "off":
                gate = gate * late.to(gate.dtype)
            elif self.early_policy == "soft":
                early = self.early_floor + (1 - self.early_floor) * soft_gate
                gate = torch.where(late, gate, early)
            elif self.early_policy == "calibrated_soft":
                if self.external_prior is None:
                    raise ValueError("external_prior is required for calibrated_soft")
                # Sharpen or flatten row evidence, then match only its batch
                # mean to the train-derived applicability prevalence.  This
                # uses no row label from the evaluation trajectory.
                evidence = soft_gate.clamp_min(1e-8).pow(self.gate_power)
                prior = torch.as_tensor(
                    self.external_prior, device=evidence.device, dtype=evidence.dtype
                ).reshape(1, 1)
                early = (evidence * prior / evidence.mean().clamp_min(1e-8)).clamp(0, 1)
                gate = torch.where(late, gate, early)
            elif self.early_policy == "topk_soft":
                if self.external_prior is None:
                    raise ValueError("external_prior is required for topk_soft")
                # Select the highest-posterior rows at the prevalence expected
                # from train.  It is a diagnostic active-set approximation,
                # not a claim that the prevalence is known in deployment.
                count = max(1, min(len(soft_gate), round(float(self.external_prior) * len(soft_gate))))
                indices = torch.topk(soft_gate[:, 0], k=count, sorted=False).indices
                early = torch.zeros_like(soft_gate)
                early[indices, 0] = 1
                gate = torch.where(late, gate, early)
            elif self.early_policy == "shuffled":
                # Same marginal active rate and timing as the hard preview,
                # but deliberately assign its row evidence to a different row.
                shuffled = torch.roll(gate, shifts=1, dims=0)
                gate = torch.where(late, gate, shuffled)
            elif self.early_policy != "gated":
                raise ValueError(f"unknown early_policy={self.early_policy}")
        if self.trace_enabled:
            self.gate_history.append(
                {
                    "t": float(t.detach().float().mean().cpu()),
                    "gate": gate[:, 0].detach().bool().cpu(),
                    "codes": codes.detach().cpu(),
                }
            )
        gated_condition = condition * gate
        self.backbone.set_condition(gated_condition)
        conditional = self.base_model(x_num, x_cat, t, sigma=sigma)
        self.backbone.set_condition(condition)
        deltas = [active - base for active, base in zip(conditional, unconditional)]
        if self.actuation_scope == "target":
            if not self.numeric_positions:
                raise ValueError("numeric_positions are required for target-scoped actuation")
            numeric_mask = torch.zeros_like(deltas[0])
            age_requested = condition[:, 2].gt(0.5)
            hours_requested = condition[:, 3].gt(0.5)
            numeric_mask[age_requested, self.numeric_positions["age"]] = 1
            numeric_mask[hours_requested, self.numeric_positions["hours.per.week"]] = 1
            deltas[0] = deltas[0] * numeric_mask
            # The current contracts only mutate numeric consequents.  Keep all
            # categorical logits unconditional so antecedent/exception columns
            # cannot be changed as an easier route to rule satisfaction.
            deltas[1] = torch.zeros_like(deltas[1])
        elif self.actuation_scope != "all":
            raise ValueError(f"unknown actuation_scope={self.actuation_scope}")
        return tuple(base + self.guidance_scale * delta for base, delta in zip(unconditional, deltas))


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype(str).str.strip()
    return result


def encoded_threshold(dataset, info: dict, frame: pd.DataFrame, column: str, threshold: float) -> float:
    num_names = [info["column_names"][index] for index in info["num_col_idx"]]
    raw = frame[num_names].median(numeric_only=True).to_numpy(dtype=float)[None, :]
    raw[0, num_names.index(column)] = threshold
    return float(dataset.num_transform.transform(raw)[0, num_names.index(column)])


def repaired_reference(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    result = frame.copy()
    violation = violation_mask(result, contract)
    result.loc[violation, contract["target"]] = contract["threshold"]
    return result


def evaluate(frame: pd.DataFrame, real: pd.DataFrame, contract: dict, baseline: dict | None = None) -> dict:
    frame, real = clean(frame), clean(real)
    antecedent = antecedent_mask(frame, contract)
    exception = exception_mask(frame, contract)
    active = antecedent & ~exception
    violation = violation_mask(frame, contract)
    real_antecedent = antecedent_mask(real, contract)
    real_exception = exception_mask(real, contract)
    real_active = real_antecedent & ~real_exception
    target = pd.to_numeric(frame[contract["target"]], errors="coerce")
    real_target = pd.to_numeric(real[contract["target"]], errors="coerce")
    reference = repaired_reference(real, contract)
    reference_target = pd.to_numeric(reference.loc[real_active, contract["target"]], errors="coerce")
    numeric = [column for column in real.columns if pd.api.types.is_numeric_dtype(real[column])]
    categoricals = [column for column in real.columns if column not in numeric]
    numeric_ks = [ks_2samp(pd.to_numeric(frame[column]), pd.to_numeric(real[column])).statistic for column in numeric]
    categorical_tv = [total_variation(frame[column], real[column]) for column in categoricals]
    target_ks = ks_2samp(target[active], reference_target).statistic if active.any() and len(reference_target) else math.nan
    real_exception_given_ant = float(real_exception.sum() / max(real_antecedent.sum(), 1))
    exception_given_ant = float(exception.sum() / max(antecedent.sum(), 1))
    result = {
        "contract_id": contract["contract_id"], "split": contract["split"],
        "antecedents": "+".join(contract["antecedents"]), "target": contract["target"],
        "threshold": contract["threshold"], "has_exception": contract["exception_exec"],
        "samples": len(frame),
        "overall_violation_rate": float(violation.mean()),
        "conditional_violation_rate": float(violation.sum() / max(active.sum(), 1)),
        "antecedent_rate": float(antecedent.mean()), "reference_antecedent_rate": float(real_antecedent.mean()),
        "antecedent_rate_error": float(abs(antecedent.mean() - real_antecedent.mean())),
        "antecedent_retention_ratio": float(antecedent.mean() / max(real_antecedent.mean(), 1e-9)),
        "exception_given_antecedent": exception_given_ant,
        "reference_exception_given_antecedent": real_exception_given_ant,
        "exception_rate_error": abs(exception_given_ant - real_exception_given_ant),
        "active_target_mean": float(target[active].mean()) if active.any() else math.nan,
        "inactive_target_mean": float(target[~antecedent].mean()) if (~antecedent).any() else math.nan,
        "reference_inactive_target_mean": float(real_target[~real_antecedent].mean()) if (~real_antecedent).any() else math.nan,
        "inactive_target_mean_error": float(abs(target[~antecedent].mean() - real_target[~real_antecedent].mean())) if (~antecedent).any() else math.nan,
        "target_boundary_mass_active": float(np.isclose(target[active], contract["threshold"], atol=0.25).mean()) if active.any() else math.nan,
        "active_target_ks_to_repaired_reference": float(target_ks),
        "global_numeric_ks": float(np.mean(numeric_ks)),
        "global_categorical_tv": float(np.mean(categorical_tv)),
    }
    if baseline:
        result["conditional_violation_reduction"] = baseline["conditional_violation_rate"] - result["conditional_violation_rate"]
        result["antecedent_error_change"] = result["antecedent_rate_error"] - baseline["antecedent_rate_error"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-mode", choices=["filter", "repair_preserve", "repair_gated"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pairs-per-contract", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--guidance-scales", default="1,2")
    parser.add_argument("--sample-steps", type=int, default=30)
    parser.add_argument("--adapter-lr", type=float, default=5e-4)
    parser.add_argument("--core-lr", type=float, default=1e-5)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    device = torch.device("cuda")
    seed_all(args.seed)
    sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    contracts = catalog(); train_contracts = [item for item in contracts if item["split"] == "train"]
    condition_dim = len(vector(contracts[0]))
    info = json.loads((args.repo / "data/adult/info.json").read_text())
    train_data, _ = preprocess_adult(args.repo)
    real = clean(decode(train_data, train_data.X, info))
    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    backbone = ConditionedUniModMLP(condition_dim=condition_dim, **params)
    base_model = Model(backbone, **config["diffusion_params"]["edm_params"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = base_model.load_state_dict(checkpoint["denoise_fn"], strict=False)
    if incompatible.unexpected_keys or [key for key in incompatible.missing_keys if "condition_projection" not in key]:
        raise RuntimeError(str(incompatible))
    if args.training_mode == "repair_gated":
        categorical_names = [info["column_names"][index] for index in info["target_col_idx"] + info["cat_col_idx"]]
        encoder_categories = train_data.cat_transform.named_steps["ordinal"].categories_
        category_codes = {
            name: {str(value).strip(): int(index) for index, value in enumerate(values)}
            for name, values in zip(categorical_names, encoder_categories)
        }
        category_positions = {name: index for index, name in enumerate(categorical_names)}
        numeric_names = [info["column_names"][index] for index in info["num_col_idx"]]
        numeric_positions = {name: index for index, name in enumerate(numeric_names)}
        conditioned = ApplicabilityGatedConditionedModel(
            base_model, backbone, condition_dim, train_data.categories, category_codes, category_positions,
            numeric_positions=numeric_positions,
        ).to(device)
    else:
        conditioned = ContractConditionedModel(base_model, backbone, condition_dim).to(device)
    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params.update(num_timesteps=args.sample_steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(
        num_classes=train_data.categories, num_numerical_features=train_data.d_numerical,
        denoise_fn=conditioned, y_only_model=None, **diffusion_params, device=device,
    ).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    for parameter in diffusion.parameters(): parameter.requires_grad_(False)
    for parameter in backbone.mlp.condition_projection.parameters(): parameter.requires_grad_(True)
    for parameter in backbone.mlp.mlp.parameters(): parameter.requires_grad_(True)

    conditioned.set_condition(torch.zeros(args.eval_samples, condition_dim, device=device), 1.0)
    seed_all(args.seed + 9000)
    with torch.no_grad(): baseline_frame = clean(decode(train_data, diffusion.sample(args.eval_samples), info))
    baselines = {item["contract_id"]: evaluate(baseline_frame, real, item) for item in contracts}

    rng = np.random.default_rng(args.seed)
    x_parts, condition_parts, pair_stats = [], [], {}
    num_names = [info["column_names"][index] for index in info["num_col_idx"]]
    for contract in train_contracts:
        violation = violation_mask(real, contract).to_numpy()
        valid = ~violation
        if args.training_mode == "filter":
            eligible = np.flatnonzero(valid)
            sampled = rng.choice(eligible, args.pairs_per_contract, replace=len(eligible) < args.pairs_per_contract)
            values = train_data.X[sampled].clone()
        else:
            sampled = rng.choice(len(real), args.pairs_per_contract, replace=False)
            values = train_data.X[sampled].clone()
            repair_rows = violation[sampled]
            if repair_rows.any():
                target_idx = num_names.index(contract["target"])
                values[torch.from_numpy(repair_rows), target_idx] = encoded_threshold(
                    train_data, info, real, contract["target"], contract["threshold"]
                )
        x_parts.append(values)
        condition_parts.append(torch.tensor(vector(contract), dtype=torch.float32).repeat(args.pairs_per_contract, 1))
        pair_stats[contract["contract_id"]] = {
            "source_antecedent_rate": float(antecedent_mask(real, contract).mean()),
            "source_conditional_violation": float(violation.sum() / max((antecedent_mask(real, contract) & ~exception_mask(real, contract)).sum(), 1)),
            "valid_rows": int(valid.sum()), "sampled": args.pairs_per_contract,
        }
    null_indices = rng.choice(len(real), args.pairs_per_contract, replace=False)
    x_parts.append(train_data.X[null_indices]); condition_parts.append(torch.zeros(args.pairs_per_contract, condition_dim))
    paired_x, paired_condition = torch.cat(x_parts), torch.cat(condition_parts)
    loader = DataLoader(TensorDataset(paired_x, paired_condition), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), pin_memory=True)
    optimizer = torch.optim.AdamW(
        [{"params": backbone.mlp.condition_projection.parameters(), "lr": args.adapter_lr},
         {"params": backbone.mlp.mlp.parameters(), "lr": args.core_lr}], weight_decay=0,
    )
    training, started = [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        diffusion.train(); dsum = csum = 0.0; count = 0
        for batch, condition in loader:
            batch, condition = batch.to(device), condition.to(device)
            drop = torch.rand(len(condition), device=device) < args.condition_dropout
            condition = condition.clone(); condition[drop] = 0
            conditioned.set_condition(condition, 1.0)
            optimizer.zero_grad(set_to_none=True)
            discrete, continuous = diffusion.mixed_loss(batch)
            (discrete + continuous).backward(); optimizer.step()
            dsum += float(discrete.detach()) * len(batch); csum += float(continuous.detach()) * len(batch); count += len(batch)
        row = {"epoch": epoch, "discrete_loss": dsum / count, "continuous_loss": csum / count,
               "elapsed_seconds": time.perf_counter() - started}
        training.append(row)
        if epoch == 1 or epoch % 10 == 0: print(json.dumps(row), flush=True)
    pd.DataFrame(training).to_csv(args.output / "training_curve.csv", index=False)

    diffusion.eval(); metric_rows = []
    conditioned.set_condition(torch.zeros(args.eval_samples, condition_dim, device=device), 1.0)
    seed_all(args.seed + 9000)
    with torch.no_grad(): null_frame = clean(decode(train_data, diffusion.sample(args.eval_samples), info))
    for item in contracts:
        null_metric = evaluate(null_frame, real, item, baselines[item["contract_id"]])
        null_metric.update(training_mode=args.training_mode, method="finetuned_null", guidance_scale=0, seed=args.seed)
        metric_rows.append(null_metric)
    # Slurm's --export uses commas as separators, so accept colon-delimited
    # lists as well (for example IMPL_GUIDANCE=1:2:4).
    for guidance in [float(value) for value in re.split(r"[:,]", args.guidance_scales) if value.strip()]:
        for ordinal, item in enumerate(contracts):
            condition = torch.tensor(vector(item), dtype=torch.float32, device=device).repeat(args.eval_samples, 1)
            conditioned.set_condition(condition, guidance)
            seed_all(args.seed + 10000 + ordinal)
            with torch.no_grad(): generated = clean(decode(train_data, diffusion.sample(args.eval_samples), info))
            result = evaluate(generated, real, item, baselines[item["contract_id"]])
            result.update(training_mode=args.training_mode, method="conditioned", guidance_scale=guidance, seed=args.seed)
            metric_rows.append(result)
            print(json.dumps({key: result[key] for key in ("contract_id", "conditional_violation_rate", "antecedent_rate_error", "exception_rate_error")}), flush=True)
    metrics = pd.DataFrame(metric_rows); metrics.to_csv(args.output / "contract_metrics.csv", index=False)
    conditioned_rows = metrics[metrics.method.eq("conditioned")]
    summary = conditioned_rows.groupby(["training_mode", "guidance_scale", "split"]).agg(
        contracts=("contract_id", "size"), conditional_violation=("conditional_violation_rate", "mean"),
        violation_reduction=("conditional_violation_reduction", "mean"), antecedent_error=("antecedent_rate_error", "mean"),
        antecedent_retention=("antecedent_retention_ratio", "mean"), exception_error=("exception_rate_error", "mean"),
        inactive_target_error=("inactive_target_mean_error", "mean"), boundary_mass=("target_boundary_mass_active", "mean"),
        global_numeric_ks=("global_numeric_ks", "mean"), global_categorical_tv=("global_categorical_tv", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary_by_split.csv", index=False)
    torch.save(
        {"condition_projection": backbone.mlp.condition_projection.state_dict(),
         "latent_mlp": backbone.mlp.mlp.state_dict(), "training_mode": args.training_mode,
         "condition_dim": condition_dim},
        args.output / "adapter.pt",
    )
    metadata = {
        "status": "executed", "training_mode": args.training_mode, "seed": args.seed,
        "epochs": args.epochs, "updates": args.epochs * len(loader), "pairs": len(paired_x),
        "pair_stats": pair_stats, "eval_samples": args.eval_samples,
        "runtime_seconds": time.perf_counter() - started, "gpu": torch.cuda.get_device_name(device),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "scope": "applicability/evasion mechanism pilot; not a paper reproduction",
        "real_data_policy": "Real train rows used locally; no real row written or sent externally.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
