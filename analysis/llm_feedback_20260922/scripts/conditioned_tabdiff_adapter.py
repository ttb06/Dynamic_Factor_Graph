#!/usr/bin/env python3
"""Fine-tune a small contract adapter inside a pretrained Adult TabDiff model.

This is a mechanism/feasibility pilot, not a paper reproduction.  It compares
request-ID, typed-contract, frozen foundation-text-embedding, and hybrid
conditions while holding the pretrained checkpoint and training pairs fixed.
Only the condition projection and TabDiff's latent MLP core are updated.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader, TensorDataset

from contract_condition_catalog import COLUMNS, catalog, satisfies, structured_vector
from tabdiff_probe import preprocess_adult


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode(dataset, encoded: torch.Tensor, info: dict) -> pd.DataFrame:
    d_num = dataset.d_numerical
    array = encoded.detach().cpu().numpy()
    num_all = dataset.int_inverse(dataset.num_inverse(array[:, :d_num])).astype(np.float32)
    cat_all = dataset.cat_inverse(array[:, d_num:])
    n_target = len(info["target_col_idx"])
    # Match TabDiff's released split_num_cat_target convention.  Regression
    # targets are prepended to the numerical block; classification targets are
    # prepended to the categorical block.  The second regression branch keeps
    # old mechanism checkpoints (which encoded the target categorically)
    # readable, but new official-protocol runs use the first branch.
    regression_target_is_numerical = (
        info["task_type"] == "regression"
        and d_num == len(info["num_col_idx"]) + n_target
    )
    if regression_target_is_numerical:
        target, num = num_all[:, :n_target], num_all[:, n_target:]
        cat = cat_all
    else:
        num = num_all
        target, cat = cat_all[:, :n_target], cat_all[:, n_target:]
    mapping = {int(key): int(value) for key, value in info["idx_mapping"].items()}
    out = pd.DataFrame(index=np.arange(len(array)))
    for original in range(len(info["column_names"])):
        name = info["column_names"][original]
        mapped = mapping[original]
        if original in info["num_col_idx"]:
            out[name] = num[:, mapped]
        elif original in info["cat_col_idx"]:
            out[name] = cat[:, mapped - len(info["num_col_idx"])]
        else:
            out[name] = target[:, mapped - len(info["num_col_idx"]) - len(info["cat_col_idx"])]
    for column in out.select_dtypes(include=["object", "string"]).columns:
        out[column] = out[column].astype(str).str.strip()
    return out


class ConditionedMLPDiffusion(nn.Module):
    """Upstream MLPDiffusion plus an additive contract embedding."""

    def __init__(self, d_in: int, condition_dim: int, dim_t: int = 512, use_mlp: bool = True):
        super().__init__()
        from tabdiff.modules.main_modules import PositionalEmbedding

        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t), nn.SiLU(),
            nn.Linear(dim_t, d_in),
        ) if use_mlp else nn.Linear(dim_t, d_in)
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))
        self.condition_projection = nn.Linear(condition_dim, dim_t, bias=False)
        nn.init.zeros_(self.condition_projection.weight)
        self.current_condition: torch.Tensor | None = None

    def set_condition(self, condition: torch.Tensor | None) -> None:
        self.current_condition = condition

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        condition = self.current_condition
        if condition is None:
            condition = torch.zeros(len(x), self.condition_projection.in_features, device=x.device, dtype=x.dtype)
        if len(condition) == 1 and len(x) != 1:
            condition = condition.expand(len(x), -1)
        if len(condition) != len(x):
            raise ValueError(f"condition batch {len(condition)} != model batch {len(x)}")
        hidden = self.proj(x) + emb + self.condition_projection(condition.to(dtype=x.dtype, device=x.device))
        return self.mlp(hidden)


class ConditionedUniModMLP(nn.Module):
    def __init__(
        self, d_numerical: int, categories: list[int], num_layers: int, d_token: int,
        condition_dim: int, n_head: int = 1, factor: int = 4, bias: bool = True,
        dim_t: int = 512, use_mlp: bool = True, **_: object,
    ):
        super().__init__()
        from tabdiff.modules.transformer import Reconstructor, Tokenizer, Transformer

        self.d_numerical = d_numerical
        self.categories = categories
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)
        self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        d_in = d_token * (d_numerical + len(categories))
        self.mlp = ConditionedMLPDiffusion(d_in, condition_dim=condition_dim, dim_t=dim_t, use_mlp=use_mlp)
        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)
        self.model = nn.ModuleList([self.tokenizer, self.encoder, self.mlp, self.decoder, self.detokenizer])

    def set_condition(self, condition: torch.Tensor | None) -> None:
        self.mlp.set_condition(condition)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, timesteps: torch.Tensor):
        encoded = self.tokenizer(x_num, x_cat)
        decoder_input = encoded[:, 1:, :]
        hidden = self.encoder(decoder_input)
        prediction = self.mlp(hidden.reshape(hidden.shape[0], -1), timesteps)
        decoded = self.decoder(prediction.reshape(*hidden.shape))
        x_num_pred, x_cat_pred = self.detokenizer(decoded)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred) else torch.zeros_like(x_cat).to(x_num_pred.dtype)
        return x_num_pred, x_cat_pred


class ContractConditionedModel(nn.Module):
    """Keep the upstream Model intact and add context plus optional CFG."""

    def __init__(self, base_model: nn.Module, backbone: ConditionedUniModMLP, condition_dim: int):
        super().__init__()
        self.base_model = base_model
        self.backbone = backbone
        self.condition_dim = condition_dim
        self.condition: torch.Tensor | None = None
        self.guidance_scale = 1.0
        self.numeric_output_indices: list[int] | None = None

    def set_condition(self, condition: torch.Tensor | None, guidance_scale: float = 1.0) -> None:
        self.condition = condition
        self.guidance_scale = float(guidance_scale)

    def set_numeric_output_mask(self, indices: list[int] | None) -> None:
        """Restrict conditional actuation to typed numeric descendants.

        ``None`` preserves the original full-record CFG behavior.  A list
        keeps every non-listed numerical output and every categorical logit on
        the unconditional path.
        """
        self.numeric_output_indices = None if indices is None else [int(index) for index in indices]

    def forward(self, x_num, x_cat, t, sigma=None):
        condition = self.condition
        if condition is None:
            condition = torch.zeros(len(x_num), self.condition_dim, device=x_num.device, dtype=x_num.dtype)
        self.backbone.set_condition(condition)
        conditional = self.base_model(x_num, x_cat, t, sigma=sigma)
        if math.isclose(self.guidance_scale, 1.0) and self.numeric_output_indices is None:
            return conditional
        self.backbone.set_condition(torch.zeros_like(condition))
        unconditional = self.base_model(x_num, x_cat, t, sigma=sigma)
        self.backbone.set_condition(condition)
        guided = tuple(base + self.guidance_scale * (active - base) for active, base in zip(conditional, unconditional))
        if self.numeric_output_indices is None:
            return guided
        numeric = unconditional[0].clone()
        if self.numeric_output_indices:
            numeric[:, self.numeric_output_indices] = guided[0][:, self.numeric_output_indices]
        return numeric, unconditional[1]


def build_vectors(representation: str, contracts: list[dict], embeddings_path: Path | None) -> tuple[dict[str, list[float]], dict]:
    train_ids = [item["contract_id"] for item in contracts if item["split"] == "train"]
    text_vectors: dict[str, list[float]] = {}
    embedding_meta: dict = {}
    if representation in {"text", "hybrid"}:
        if embeddings_path is None:
            raise ValueError("text/hybrid representation requires --embeddings")
        embedding_payload = json.loads(embeddings_path.read_text())
        text_vectors = embedding_payload["vectors"]
        embedding_meta = {key: embedding_payload.get(key) for key in ("model", "dimensions", "usage")}
    vectors: dict[str, list[float]] = {}
    for contract in contracts:
        if representation == "structured":
            vector = structured_vector(contract)
        elif representation == "request_id":
            vector = [0.0] * len(train_ids)
            if contract["contract_id"] in train_ids:
                vector[train_ids.index(contract["contract_id"])] = 1.0
        elif representation == "text":
            vector = list(text_vectors[contract["contract_id"]])
        elif representation == "hybrid":
            vector = structured_vector(contract) + list(text_vectors[contract["contract_id"]])
        else:
            raise ValueError(representation)
        array = np.asarray(vector, dtype=np.float32)
        if representation in {"text", "hybrid"}:
            # Normalize only the foundation embedding suffix; keep typed values
            # at their meaningful scale in the hybrid condition.
            prefix = len(structured_vector(contract)) if representation == "hybrid" else 0
            norm = np.linalg.norm(array[prefix:])
            if norm > 0:
                array[prefix:] /= norm
        vectors[contract["contract_id"]] = array.tolist()
    return vectors, {"train_request_ids": train_ids, "embedding": embedding_meta}


def contract_mask(frame: pd.DataFrame, contract: dict) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for predicate in contract["predicates"]:
        values = pd.to_numeric(frame[predicate["column"]], errors="coerce")
        mask &= values.ge(float(predicate["threshold"]))
    return mask


def total_variation(sample: pd.Series, reference: pd.Series) -> float:
    left = sample.astype(str).value_counts(normalize=True)
    right = reference.astype(str).value_counts(normalize=True)
    support = left.index.union(right.index)
    return float(0.5 * (left.reindex(support, fill_value=0) - right.reindex(support, fill_value=0)).abs().sum())


def metrics_for(sample: pd.DataFrame, reference_all: pd.DataFrame, contract: dict, baseline_rate: float) -> dict:
    intended = contract_mask(sample, contract)
    reference = reference_all[contract_mask(reference_all, contract)]
    num_cols = [column for column in reference_all.columns if pd.api.types.is_numeric_dtype(reference_all[column])]
    cat_cols = [column for column in reference_all.columns if column not in num_cols]
    cohort_ks, marginal_ks = [], []
    for column in num_cols:
        observed = pd.to_numeric(sample[column], errors="coerce").dropna()
        cohort = pd.to_numeric(reference[column], errors="coerce").dropna()
        marginal = pd.to_numeric(reference_all[column], errors="coerce").dropna()
        if len(observed) and len(cohort):
            cohort_ks.append(float(ks_2samp(observed, cohort).statistic))
        if len(observed) and len(marginal):
            marginal_ks.append(float(ks_2samp(observed, marginal).statistic))
    cohort_tv = [total_variation(sample[column], reference[column]) for column in cat_cols if len(reference)]
    rare_coverages = []
    for column in cat_cols:
        counts = reference[column].astype(str).value_counts()
        required = set(counts[counts >= max(5, int(0.005 * len(reference)))].index)
        if required:
            rare_coverages.append(len(required & set(sample[column].astype(str))) / len(required))
    shortfall = []
    for predicate in contract["predicates"]:
        values = pd.to_numeric(sample[predicate["column"]], errors="coerce")
        shortfall.append(np.maximum(float(predicate["threshold"]) - values, 0).mean())
    rounded = sample.copy()
    for column in num_cols:
        rounded[column] = pd.to_numeric(rounded[column], errors="coerce").round(2)
    return {
        "contract_id": contract["contract_id"],
        "split": contract["split"],
        "family": contract["family"],
        "predicates": json.dumps(contract["predicates"]),
        "samples": len(sample),
        "satisfaction_rate": float(intended.mean()),
        "violation_rate": float(1 - intended.mean()),
        "unconditional_satisfaction_rate": float(baseline_rate),
        "satisfaction_gain": float(intended.mean() - baseline_rate),
        "mean_shortfall": float(np.mean(shortfall)),
        "cohort_numeric_ks_mean": float(np.mean(cohort_ks)) if cohort_ks else None,
        "marginal_numeric_ks_mean": float(np.mean(marginal_ks)) if marginal_ks else None,
        "cohort_categorical_tv_mean": float(np.mean(cohort_tv)) if cohort_tv else None,
        "cohort_category_coverage_mean": float(np.mean(rare_coverages)) if rare_coverages else None,
        "duplicate_rate_rounded": float(rounded.duplicated().mean()),
        "reference_cohort_size": len(reference),
        "target_means": json.dumps({column: float(pd.to_numeric(sample[column], errors="coerce").mean()) for column in COLUMNS}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=["request_id", "structured", "text", "hybrid"], required=True)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--pairs-per-contract", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--adapter-lr", type=float, default=5e-4)
    parser.add_argument("--core-lr", type=float, default=1e-5)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--text-branch-dropout", type=float, default=0.0,
                        help="For hybrid conditions, independently zero the frozen-text suffix during training.")
    parser.add_argument("--monotonic-weight", type=float, default=0.0,
                        help="Weight for paired same-noise low/high-condition ordering loss.")
    parser.add_argument("--monotonic-margin", type=float, default=0.05,
                        help="Minimum encoded x0 increase requested by the ordering loss.")
    parser.add_argument("--monotonic-batch", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--guidance-scales", default="1,2")
    parser.add_argument("--sample-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("conditional adapter pilot requires a GPU")
    device = torch.device("cuda")
    seed_all(args.seed)
    sys.path.insert(0, str(args.repo))

    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model

    contracts = catalog()
    train_contracts = [item for item in contracts if item["split"] == "train"]
    vectors, vector_meta = build_vectors(args.representation, contracts, args.embeddings)
    condition_dim = len(next(iter(vectors.values())))
    info = json.loads((args.repo / "data/adult/info.json").read_text())
    train_data, _ = preprocess_adult(args.repo)
    real_frame = decode(train_data, train_data.X, info)

    with args.config.open("rb") as handle:
        config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    backbone = ConditionedUniModMLP(condition_dim=condition_dim, **params)
    base_model = Model(backbone, **config["diffusion_params"]["edm_params"])
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    incompatible = base_model.load_state_dict(checkpoint["denoise_fn"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if "condition_projection" not in key]
    if unexpected or missing:
        raise RuntimeError(f"checkpoint mismatch; unexpected={unexpected}, missing={missing}")
    conditioned_model = ContractConditionedModel(base_model, backbone, condition_dim).to(device)

    diffusion_params = copy.deepcopy(config["diffusion_params"])
    diffusion_params["num_timesteps"] = args.sample_steps
    diffusion_params["scheduler"] = "power_mean_per_column"
    diffusion_params["cat_scheduler"] = "log_linear_per_column"
    diffusion_params["sampler_params"] = copy.deepcopy(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"]["stochastic_sampler"] = False
    diffusion_params["sampler_params"]["second_order_correction"] = False
    diffusion = UnifiedCtimeDiffusion(
        num_classes=train_data.categories,
        num_numerical_features=train_data.d_numerical,
        denoise_fn=conditioned_model,
        y_only_model=None,
        **diffusion_params,
        device=device,
    ).to(device)
    diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"])
    diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"])
    for parameter in diffusion.parameters():
        parameter.requires_grad_(False)
    for parameter in backbone.mlp.condition_projection.parameters():
        parameter.requires_grad_(True)
    for parameter in backbone.mlp.mlp.parameters():
        parameter.requires_grad_(True)

    # A true frozen-checkpoint baseline before any update.  One sample is reused
    # to score all requests, avoiding an artificial per-request lottery.
    conditioned_model.set_condition(torch.zeros(args.eval_samples, condition_dim, device=device), 1.0)
    seed_all(args.seed + 9000)
    with torch.no_grad():
        baseline_encoded = diffusion.sample(args.eval_samples)
    baseline_frame = decode(train_data, baseline_encoded, info)
    baseline_rates = {item["contract_id"]: float(contract_mask(baseline_frame, item).mean()) for item in contracts}

    rng = np.random.default_rng(args.seed)
    pair_x, pair_conditions, pair_counts = [], [], {}
    for contract in train_contracts:
        eligible = np.flatnonzero(contract_mask(real_frame, contract).to_numpy())
        if not len(eligible):
            raise RuntimeError(f"no rows satisfy {contract['contract_id']}")
        sampled = rng.choice(eligible, size=args.pairs_per_contract, replace=len(eligible) < args.pairs_per_contract)
        pair_x.append(train_data.X[sampled])
        vector = torch.tensor(vectors[contract["contract_id"]], dtype=torch.float32)
        pair_conditions.append(vector.repeat(args.pairs_per_contract, 1))
        pair_counts[contract["contract_id"]] = {"eligible": int(len(eligible)), "sampled": args.pairs_per_contract}
    null_indices = rng.choice(len(train_data.X), size=args.pairs_per_contract, replace=False)
    pair_x.append(train_data.X[null_indices])
    pair_conditions.append(torch.zeros(args.pairs_per_contract, condition_dim))
    paired_x = torch.cat(pair_x)
    paired_conditions = torch.cat(pair_conditions)
    loader = DataLoader(
        TensorDataset(paired_x, paired_conditions), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0, pin_memory=True,
    )

    # Training-only pairs use values that occur in the training catalog.  This
    # is an auxiliary semantic-consistency objective, not supervision on the
    # held-out thresholds used by the paired-noise evaluation.
    monotonic_specs = [
        ("age", "age_ge_25", "age_ge_50"),
        ("hours.per.week", "hours_ge_20", "hours_ge_50"),
        ("education.num", "edu_ge_9", "edu_ge_14"),
    ]
    num_names = [info["column_names"][index] for index in info["num_col_idx"]]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone.mlp.condition_projection.parameters(), "lr": args.adapter_lr},
            {"params": backbone.mlp.mlp.parameters(), "lr": args.core_lr},
        ],
        weight_decay=0,
    )
    training_rows = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        total_d, total_c, total_m, count = 0.0, 0.0, 0.0, 0
        for batch, condition in loader:
            batch = batch.to(device, non_blocking=True)
            condition = condition.to(device, non_blocking=True)
            if args.condition_dropout:
                drop = torch.rand(len(condition), device=device) < args.condition_dropout
                condition = condition.clone(); condition[drop] = 0
            if args.representation == "hybrid" and args.text_branch_dropout:
                text_drop = torch.rand(len(condition), device=device) < args.text_branch_dropout
                condition = condition.clone(); condition[text_drop, 6:] = 0
            conditioned_model.set_condition(condition, 1.0)
            optimizer.zero_grad(set_to_none=True)
            discrete, continuous = diffusion.mixed_loss(batch)
            monotonic = torch.zeros((), device=device)
            if args.monotonic_weight > 0:
                n_order = min(len(batch), args.monotonic_batch)
                clean_num = batch[:n_order, :train_data.d_numerical]
                clean_cat = batch[:n_order, train_data.d_numerical:].long()
                order_t = 0.1 + 0.8 * torch.rand(n_order, 1, device=device, dtype=clean_num.dtype)
                order_sigma = diffusion.num_schedule.total_noise(order_t)
                noisy_num = clean_num + torch.randn_like(clean_num) * order_sigma
                clean_cat_one_hot = diffusion.to_one_hot(clean_cat).to(clean_num.dtype)
                per_family = []
                for target, low_id, high_id in monotonic_specs:
                    low = torch.tensor(vectors[low_id], device=device, dtype=clean_num.dtype).repeat(n_order, 1)
                    high = torch.tensor(vectors[high_id], device=device, dtype=clean_num.dtype).repeat(n_order, 1)
                    conditioned_model.set_condition(low, 1.0)
                    low_x0, _ = conditioned_model(noisy_num, clean_cat_one_hot, order_t.squeeze(-1), sigma=order_sigma)
                    conditioned_model.set_condition(high, 1.0)
                    high_x0, _ = conditioned_model(noisy_num, clean_cat_one_hot, order_t.squeeze(-1), sigma=order_sigma)
                    target_index = num_names.index(target)
                    per_family.append(torch.relu(args.monotonic_margin - (high_x0[:, target_index] - low_x0[:, target_index])).mean())
                monotonic = torch.stack(per_family).mean()
                conditioned_model.set_condition(condition, 1.0)
            loss = discrete + continuous + args.monotonic_weight * monotonic
            loss.backward()
            optimizer.step()
            total_d += float(discrete.detach()) * len(batch)
            total_c += float(continuous.detach()) * len(batch)
            total_m += float(monotonic.detach()) * len(batch)
            count += len(batch)
        row = {
            "epoch": epoch, "discrete_loss": total_d / count, "continuous_loss": total_c / count,
            "monotonic_loss": total_m / count,
            "elapsed_seconds": time.perf_counter() - started,
        }
        training_rows.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(row), flush=True)
    pd.DataFrame(training_rows).to_csv(args.output / "training_curve.csv", index=False)

    # Evaluate null drift and every seen/unseen contract.  Same seed per
    # contract gives paired initial noise across representations/scales.
    diffusion.eval()
    conditioned_model.set_condition(torch.zeros(args.eval_samples, condition_dim, device=device), 1.0)
    seed_all(args.seed + 9000)
    with torch.no_grad():
        finetuned_null = decode(train_data, diffusion.sample(args.eval_samples), info)
    null_rates = {item["contract_id"]: float(contract_mask(finetuned_null, item).mean()) for item in contracts}

    metric_rows, samples = [], []
    for guidance in [float(value) for value in args.guidance_scales.split(",") if value.strip()]:
        for ordinal, contract in enumerate(contracts):
            condition = torch.tensor(vectors[contract["contract_id"]], device=device).float().repeat(args.eval_samples, 1)
            conditioned_model.set_condition(condition, guidance)
            seed_all(args.seed + 10000 + ordinal)
            with torch.no_grad():
                generated = decode(train_data, diffusion.sample(args.eval_samples), info)
            metrics = metrics_for(generated, real_frame, contract, baseline_rates[contract["contract_id"]])
            metrics.update(
                {
                    "representation": args.representation,
                    "guidance_scale": guidance,
                    "finetuned_null_satisfaction_rate": null_rates[contract["contract_id"]],
                    "gain_over_finetuned_null": metrics["satisfaction_rate"] - null_rates[contract["contract_id"]],
                    "condition_norm": float(np.linalg.norm(vectors[contract["contract_id"]])),
                }
            )
            metric_rows.append(metrics)
            generated.insert(0, "contract_id", contract["contract_id"])
            generated.insert(1, "contract_split", contract["split"])
            generated.insert(2, "guidance_scale", guidance)
            generated.insert(3, "privacy", "synthetic_generated")
            samples.append(generated)
            print(json.dumps({key: metrics[key] for key in ("contract_id", "split", "guidance_scale", "satisfaction_rate", "satisfaction_gain")}), flush=True)

    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(args.output / "contract_metrics.csv", index=False)
    pd.concat(samples, ignore_index=True).to_csv(args.output / "synthetic_samples.csv", index=False)
    summary_by_split = (
        metric_frame.groupby(["representation", "guidance_scale", "split"])
        .agg(
            contracts=("contract_id", "size"), satisfaction_rate=("satisfaction_rate", "mean"),
            satisfaction_gain=("satisfaction_gain", "mean"), gain_over_finetuned_null=("gain_over_finetuned_null", "mean"),
            cohort_numeric_ks=("cohort_numeric_ks_mean", "mean"), cohort_categorical_tv=("cohort_categorical_tv_mean", "mean"),
            category_coverage=("cohort_category_coverage_mean", "mean"), duplicate_rate=("duplicate_rate_rounded", "mean"),
        ).reset_index()
    )
    summary_by_split.to_csv(args.output / "summary_by_split.csv", index=False)
    torch.save(
        {
            "condition_projection": backbone.mlp.condition_projection.state_dict(),
            "latent_mlp": backbone.mlp.mlp.state_dict(),
            "representation": args.representation,
            "condition_dim": condition_dim,
            "vectors": vectors,
        },
        args.output / "adapter.pt",
    )
    metadata = {
        "status": "executed",
        "scope": "conditional adapter feasibility pilot; not a paper reproduction",
        "representation": args.representation,
        "condition_dim": condition_dim,
        "epochs": args.epochs,
        "updates": args.epochs * len(loader),
        "training_pairs": len(paired_x),
        "pair_counts": pair_counts,
        "condition_dropout": args.condition_dropout,
        "text_branch_dropout": args.text_branch_dropout,
        "monotonic_weight": args.monotonic_weight,
        "monotonic_margin": args.monotonic_margin,
        "monotonic_batch": args.monotonic_batch,
        "sample_steps": args.sample_steps,
        "eval_samples_per_contract": args.eval_samples,
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(device),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "checkpoint": str(args.checkpoint),
        "vector_metadata": vector_meta,
        "real_data_policy": "Real train rows were used locally for fitting/evaluation; no real rows are written or sent to an API.",
        "synthetic_output": "synthetic_samples.csv contains generated rows only.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
