#!/usr/bin/env python3
"""Small, reproducible TabDiff trajectory and intervention study.

This is intentionally a mechanism pilot.  It reuses the official TabDiff model
and preprocessing, but the guidance/repair methods implemented here are clearly
named proxies rather than claimed reproductions of Harpoon, SCTab-Diff, or LCATS.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # official environment may be preprocessing-only
    plt = None
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler


def add_repo_to_path(repo: Path) -> None:
    sys.path.insert(0, str(repo))


@dataclass
class RuleContext:
    education_map: dict[str, float]


@dataclass
class ModelBundle:
    diffusion: object
    dataset_train: object
    dataset_test: object
    info: dict
    device: torch.device
    cat_names: list[str]
    num_names: list[str]
    cat_name_to_idx: dict[str, int]
    num_name_to_idx: dict[str, int]
    category_codes: dict[str, dict[str, int]]
    education_target_by_code: torch.Tensor


class AnalysisDataset:
    """Minimal reproduction of TabDiff's Adult preprocessing.

    Keeping this local avoids importing evaluation-only dependencies from the
    upstream ``src`` package in the GPU environment.
    """

    def __init__(self, x: torch.Tensor, num_transform, cat_transform, categories: np.ndarray,
                 d_numerical: int, int_inverse=None):
        self.X = x
        self.num_transform = num_transform
        self.cat_transform = cat_transform
        self._int_inverse = int_inverse
        self.categories = categories
        self.d_numerical = d_numerical

    def __len__(self):
        return len(self.X)

    def num_inverse(self, values):
        if hasattr(self.num_transform, "inverse_transform"):
            return self.num_transform.inverse_transform(values)
        return self.num_transform(values)

    def int_inverse(self, values):
        return self._int_inverse(values) if self._int_inverse is not None else values

    def cat_inverse(self, values):
        if not hasattr(self.cat_transform, "named_steps"):
            return self.cat_transform(values)
        array = np.asarray(values).copy()
        encoder = self.cat_transform.named_steps["ordinal"]
        invalid = np.zeros(array.shape, dtype=bool)
        safe = array.copy()
        for column, categories in enumerate(encoder.categories_):
            column_invalid = (~np.isfinite(safe[:, column])) | (safe[:, column] < 0) | (safe[:, column] >= len(categories))
            invalid[:, column] = column_invalid
            safe[column_invalid, column] = 0
        decoded = self.cat_transform.inverse_transform(safe).astype(object)
        decoded[invalid] = "<UNKNOWN>"
        return decoded


def preprocess_dataset(repo: Path, dataset: str,
                       regression_target_mode: str = "categorical",
                       ) -> tuple[AnalysisDataset, AnalysisDataset]:
    """Reproduce the local TabDiff preprocessing for any prepared dataset.

    The prepared Adult, Default, and Shoppers directories share the same
    X_num/X_cat/y layout.  Keeping one train-fitted transform here also avoids
    accidental held-out leakage when adding cross-dataset pilots.
    """
    data = repo / "data" / dataset
    info = json.loads((data / "info.json").read_text())
    if regression_target_mode not in {"categorical", "numerical"}:
        raise ValueError("regression_target_mode must be categorical or numerical")
    x_num_train = np.load(data / "X_num_train.npy", allow_pickle=True).astype(float)
    x_num_test = np.load(data / "X_num_test.npy", allow_pickle=True).astype(float)
    x_cat_train = np.load(data / "X_cat_train.npy", allow_pickle=True).astype(str)
    x_cat_test = np.load(data / "X_cat_test.npy", allow_pickle=True).astype(str)
    if info["task_type"] == "regression" and regression_target_mode == "numerical":
        # Exact released convention: regression y is the first numerical
        # feature.  The transform parameters below mirror src.Transformations.
        y_train = np.load(data / "y_train.npy", allow_pickle=True).reshape(-1, 1).astype(float)
        y_test = np.load(data / "y_test.npy", allow_pickle=True).reshape(-1, 1).astype(float)
        x_num_train = np.concatenate([y_train, x_num_train], axis=1)
        x_num_test = np.concatenate([y_test, x_num_test], axis=1)
    else:
        # Backward-compatible mode for already trained mechanism checkpoints.
        y_train = np.load(data / "y_train.npy", allow_pickle=True).reshape(-1, 1).astype(str)
        y_test = np.load(data / "y_test.npy", allow_pickle=True).reshape(-1, 1).astype(str)
        x_cat_train = np.concatenate([y_train, x_cat_train], axis=1)
        x_cat_test = np.concatenate([y_test, x_cat_test], axis=1)

    means = np.nanmean(x_num_train, axis=0)
    for array in (x_num_train, x_num_test):
        where = np.where(np.isnan(array))
        array[where] = np.take(means, where[1])
    num_transform = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=max(min(len(x_num_train) // 30, 1000), 10),
        subsample=int(1e9),
        random_state=0,
    ).fit(x_num_train)
    train_num = num_transform.transform(x_num_train)
    test_num = num_transform.transform(x_num_test)

    unknown = np.iinfo("int64").max - 3
    cat_transform = Pipeline(
        [("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=unknown, dtype="int64"))]
    ).fit(x_cat_train)
    train_cat = cat_transform.transform(x_cat_train)
    test_cat = cat_transform.transform(x_cat_test)
    max_values = train_cat.max(axis=0)
    for column in range(test_cat.shape[1]):
        test_cat[test_cat[:, column] == unknown, column] = max_values[column] + 1
    categories = np.array([len(set(train_cat[:, i].tolist())) for i in range(train_cat.shape[1])])
    train_tensor = torch.tensor(np.concatenate([train_num, train_cat], axis=1)).float()
    test_tensor = torch.tensor(np.concatenate([test_num, test_cat], axis=1)).float()
    return (
        AnalysisDataset(train_tensor, num_transform, cat_transform, categories, train_num.shape[1]),
        AnalysisDataset(test_tensor, num_transform, cat_transform, categories, train_num.shape[1]),
    )


def preprocess_adult(repo: Path) -> tuple[AnalysisDataset, AnalysisDataset]:
    """Backward-compatible alias used by the original Adult probes."""
    return preprocess_dataset(repo, "adult")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(repo: Path, checkpoint: Path, config_path: Path, sample_steps: int, device: torch.device) -> ModelBundle:
    add_repo_to_path(repo)
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    info = json.loads((repo / "data/adult/info.json").read_text())
    with config_path.open("rb") as handle:
        raw_config = pickle.load(handle)
    train_data, test_data = preprocess_adult(repo)
    params = dict(raw_config["unimodmlp_params"])
    params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    backbone = UniModMLP(**params)
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"]).to(device)

    diffusion_params = dict(raw_config["diffusion_params"])
    diffusion_params["num_timesteps"] = sample_steps
    # Official main.py switches to feature-wise schedules after model construction.
    diffusion_params["scheduler"] = "power_mean_per_column"
    diffusion_params["cat_scheduler"] = "log_linear_per_column"
    diffusion_params["sampler_params"] = dict(diffusion_params["sampler_params"])
    diffusion_params["sampler_params"]["stochastic_sampler"] = False
    diffusion_params["sampler_params"]["second_order_correction"] = False
    diffusion = UnifiedCtimeDiffusion(
        num_classes=train_data.categories,
        num_numerical_features=train_data.d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **diffusion_params,
        device=device,
    ).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()

    num_names = [info["column_names"][i] for i in info["num_col_idx"]]
    target_names = [info["column_names"][i] for i in info["target_col_idx"]]
    cat_feature_names = [info["column_names"][i] for i in info["cat_col_idx"]]
    cat_names = target_names + cat_feature_names
    cat_name_to_idx = {name: i for i, name in enumerate(cat_names)}
    num_name_to_idx = {name: i for i, name in enumerate(num_names)}

    category_codes: dict[str, dict[str, int]] = {}
    cat_encoder = train_data.cat_transform.named_steps["ordinal"]
    for name, values in zip(cat_names, cat_encoder.categories_):
        category_codes[name] = {str(v).strip(): int(i) for i, v in enumerate(values)}

    edu_cat_idx = cat_name_to_idx["education"]
    edu_num_idx = num_name_to_idx["education.num"]
    target_by_code = torch.full((int(train_data.categories[edu_cat_idx]),), float("nan"), dtype=torch.float32)
    train_encoded = train_data.X
    for code in range(len(target_by_code)):
        values = train_encoded[train_encoded[:, train_data.d_numerical + edu_cat_idx].long() == code, edu_num_idx]
        if len(values):
            target_by_code[code] = values.median()

    return ModelBundle(
        diffusion=diffusion,
        dataset_train=train_data,
        dataset_test=test_data,
        info=info,
        device=device,
        cat_names=cat_names,
        num_names=num_names,
        cat_name_to_idx=cat_name_to_idx,
        num_name_to_idx=num_name_to_idx,
        category_codes=category_codes,
        education_target_by_code=target_by_code.to(device),
    )


def decode_tensor(bundle: ModelBundle, encoded: torch.Tensor) -> pd.DataFrame:
    info = bundle.info
    d_num = bundle.dataset_train.d_numerical
    array = encoded.detach().cpu().numpy()
    syn_num = bundle.dataset_train.num_inverse(array[:, :d_num]).astype(np.float32)
    syn_num = bundle.dataset_train.int_inverse(syn_num).astype(np.float32)
    syn_cat_all = bundle.dataset_train.cat_inverse(array[:, d_num:])
    syn_target = syn_cat_all[:, :1]
    syn_cat = syn_cat_all[:, 1:]
    mapping = {int(k): int(v) for k, v in info["idx_mapping"].items()}
    out = pd.DataFrame(index=np.arange(len(array)))
    for original_idx in range(len(info["column_names"])):
        if original_idx in info["num_col_idx"]:
            out[info["column_names"][original_idx]] = syn_num[:, mapping[original_idx]]
        elif original_idx in info["cat_col_idx"]:
            out[info["column_names"][original_idx]] = syn_cat[:, mapping[original_idx] - len(info["num_col_idx"])]
        else:
            out[info["column_names"][original_idx]] = syn_target[:, mapping[original_idx] - len(info["num_col_idx"]) - len(info["cat_col_idx"])]
    return out


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.select_dtypes(include=["object", "string"]).columns:
        result[col] = result[col].astype(str).str.strip()
    return result


def make_rule_context(real_train: pd.DataFrame) -> RuleContext:
    train = clean_strings(real_train)
    mapping = train.groupby("education")["education.num"].agg(lambda s: float(s.value_counts().index[0])).to_dict()
    return RuleContext(education_map=mapping)


def rule_flags(df: pd.DataFrame, context: RuleContext) -> pd.DataFrame:
    data = clean_strings(df)
    expected_edu = data["education"].map(context.education_map)
    flags = pd.DataFrame(index=data.index)
    flags["education_fd"] = expected_edu.notna() & ~np.isclose(pd.to_numeric(data["education.num"], errors="coerce"), expected_edu, equal_nan=False)
    flags["husband_male"] = data["relationship"].eq("Husband") & ~data["sex"].eq("Male")
    flags["wife_female"] = data["relationship"].eq("Wife") & ~data["sex"].eq("Female")
    flags["spouse_marital"] = data["relationship"].isin(["Husband", "Wife"]) & ~data["marital.status"].eq("Married-civ-spouse")
    flags["capital_gain_nonnegative"] = pd.to_numeric(data["capital.gain"], errors="coerce").lt(0)
    flags["capital_loss_nonnegative"] = pd.to_numeric(data["capital.loss"], errors="coerce").lt(0)
    flags["hours_positive"] = pd.to_numeric(data["hours.per.week"], errors="coerce").lt(1)
    flags["any_hard"] = flags.any(axis=1)
    return flags


def x0_categorical_prediction(diffusion, x_cat: torch.Tensor, raw_logits: torch.Tensor) -> torch.Tensor:
    chunks = torch.split(raw_logits, diffusion.num_classes_w_mask, dim=-1)
    predicted = torch.stack([chunk[:, : int(k)].argmax(dim=-1) for chunk, k in zip(chunks, diffusion.num_classes)], dim=1)
    return torch.where(x_cat != diffusion.mask_index[None, :], x_cat, predicted)


def model_preview(bundle: ModelBundle, x_num: torch.Tensor, x_cat: torch.Tensor, t_scalar: torch.Tensor, sigma_num: torch.Tensor, requires_grad=False):
    diffusion = bundle.diffusion
    batch = len(x_num)
    x_input = x_num.detach().requires_grad_(requires_grad)
    cat_oh = diffusion.to_one_hot(x_cat).to(x_input.dtype)
    denoised, raw_logits = diffusion._denoise_fn(
        x_input.float(),
        cat_oh,
        t_scalar.squeeze().repeat(batch),
        sigma=sigma_num.unsqueeze(0).repeat(batch, 1),
    )
    cat_pred = x0_categorical_prediction(diffusion, x_cat, raw_logits)
    return x_input, denoised, raw_logits, cat_pred


def apply_typed_prediction_correction(bundle: ModelBundle, denoised: torch.Tensor, cat_pred: torch.Tensor, x_cat_next: torch.Tensor, only_violations=True):
    edu_cat_idx = bundle.cat_name_to_idx["education"]
    edu_num_idx = bundle.num_name_to_idx["education.num"]
    edu_codes = cat_pred[:, edu_cat_idx].long().clamp(0, len(bundle.education_target_by_code) - 1)
    targets = bundle.education_target_by_code[edu_codes]
    valid = torch.isfinite(targets)
    if only_violations:
        valid &= (denoised[:, edu_num_idx] - targets).abs() > 0.02
    corrected_num = denoised.clone()
    corrected_num[valid, edu_num_idx] = targets[valid]

    corrected_cat = x_cat_next.clone()
    relation_idx = bundle.cat_name_to_idx["relationship"]
    sex_idx = bundle.cat_name_to_idx["sex"]
    marital_idx = bundle.cat_name_to_idx["marital.status"]
    rel_map = bundle.category_codes.get("relationship", {})
    sex_map = bundle.category_codes.get("sex", {})
    marital_map = bundle.category_codes.get("marital.status", {})
    spouse_code = marital_map.get("Married-civ-spouse")
    for relation, desired_sex in (("Husband", "Male"), ("Wife", "Female")):
        if relation not in rel_map or desired_sex not in sex_map:
            continue
        rows = cat_pred[:, relation_idx].eq(rel_map[relation])
        if only_violations:
            rows &= cat_pred[:, sex_idx].ne(sex_map[desired_sex]) | (
                cat_pred[:, marital_idx].ne(spouse_code) if spouse_code is not None else False
            )
        corrected_cat[rows, sex_idx] = sex_map[desired_sex]
        if spouse_code is not None:
            corrected_cat[rows, marital_idx] = spouse_code
    return corrected_num, corrected_cat, int(valid.sum().item())


def sampler_grid(diffusion, steps: int, device: torch.device):
    t = torch.linspace(0, 1, steps, dtype=torch.float32, device=device)[:, None]
    sigma_num = diffusion.num_schedule.total_noise(t)
    sigma_cat = diffusion.cat_schedule.total_noise(t)
    sigma_num_next = torch.zeros_like(sigma_num)
    sigma_num_next[1:] = sigma_num[:-1]
    sigma_cat_next = torch.zeros_like(sigma_cat)
    sigma_cat_next[1:] = sigma_cat[:-1]
    return t, sigma_num, sigma_cat, sigma_num_next, sigma_cat_next


def sample_variant(bundle: ModelBundle, n: int, steps: int, seed: int, variant: str, intervention_t: float, harpoon_eta: float, collect_trace=False):
    diffusion = bundle.diffusion
    device = bundle.device
    seed_everything(seed)
    t, sigma_num, sigma_cat, sigma_num_next, sigma_cat_next = sampler_grid(diffusion, steps, device)
    x_num = torch.randn((n, diffusion.num_numerical_features), device=device) * sigma_num[-1]
    x_cat = diffusion._sample_masked_prior(n, len(diffusion.num_classes))
    trace = []
    feedback_cases = []
    corrections = 0
    intervention_i = int(torch.argmin((t.squeeze() - intervention_t).abs()).item())
    synchronize(device)
    start = time.perf_counter()

    for i in reversed(range(steps)):
        needs_grad = variant == "harpoon_numeric_proxy" and float(t[i]) <= 0.75
        context = torch.enable_grad() if needs_grad else torch.no_grad()
        with context:
            x_input, denoised, raw_logits, cat_pred = model_preview(bundle, x_num, x_cat, t[i], sigma_num[i], requires_grad=needs_grad)
            preview_encoded = torch.cat([denoised.detach(), cat_pred.detach().float()], dim=1)
            if collect_trace:
                preview_df = decode_tensor(bundle, preview_encoded)
                preview_flags = rule_flags(preview_df, sample_variant.rule_context)
                trace.append(
                    {
                        "step_index": i,
                        "t": float(t[i].item()),
                        "sigma_num_mean": float(sigma_num[i].mean().item()),
                        "masked_cat_fraction": float((x_cat == diffusion.mask_index[None, :]).float().mean().item()),
                        "preview_rule_violation_rate": float(preview_flags["any_hard"].mean()),
                        "preview_encoded": preview_encoded.cpu(),
                    }
                )
                if min(abs(float(t[i]) - target) for target in (0.7, 0.4, 0.1)) < 0.03:
                    for row_idx in range(min(12, len(preview_df))):
                        record = {}
                        for key, value in preview_df.iloc[row_idx].to_dict().items():
                            record[key] = value.item() if isinstance(value, np.generic) else value
                        feedback_cases.append(
                            {
                                "case_id": f"t{float(t[i]):.3f}_r{row_idx}",
                                "t": float(t[i]),
                                "record": record,
                                "verifier_violations": [
                                    col for col in preview_flags.columns if col != "any_hard" and bool(preview_flags.iloc[row_idx][col])
                                ],
                            }
                        )

            # Standard deterministic Euler / absorbing categorical update.
            denoised_for_step = denoised
            d_cur = (x_input - denoised_for_step) / sigma_num[i]
            x_num_next = x_input + (sigma_num_next[i] - sigma_num[i]) * d_cur
            logits = diffusion._subs_parameterization(raw_logits, x_cat)
            alpha_t = torch.exp(-sigma_cat[i]).unsqueeze(0).repeat(n, 1)
            alpha_s = torch.exp(-sigma_cat_next[i]).unsqueeze(0).repeat(n, 1)
            x_cat_next, _ = diffusion._mdlm_update(logits, x_cat, alpha_t, alpha_s)

            if variant == "fixed_rule_denoising_proxy" and float(t[i]) <= 0.75:
                corrected_prediction, corrected_cat, changed = apply_typed_prediction_correction(
                    bundle, denoised, cat_pred, x_cat_next, only_violations=True
                )
                corrections += changed
                d_cur = (x_input - corrected_prediction) / sigma_num[i]
                x_num_next = x_input + (sigma_num_next[i] - sigma_num[i]) * d_cur
                x_cat_next = corrected_cat
            elif variant == "dynamic_record_feedback_proxy" and i == intervention_i:
                corrected_prediction, corrected_cat, changed = apply_typed_prediction_correction(
                    bundle, denoised, cat_pred, x_cat_next, only_violations=True
                )
                corrections += changed
                d_cur = (x_input - corrected_prediction) / sigma_num[i]
                x_num_next = x_input + (sigma_num_next[i] - sigma_num[i]) * d_cur
                x_cat_next = corrected_cat
            elif needs_grad:
                edu_cat_idx = bundle.cat_name_to_idx["education"]
                edu_num_idx = bundle.num_name_to_idx["education.num"]
                edu_codes = cat_pred[:, edu_cat_idx].long().clamp(0, len(bundle.education_target_by_code) - 1)
                targets = bundle.education_target_by_code[edu_codes]
                valid = torch.isfinite(targets)
                loss = torch.where(valid, (denoised[:, edu_num_idx] - targets).square(), torch.zeros_like(targets))
                grad = torch.autograd.grad(loss.sum(), x_input, retain_graph=False, create_graph=False)[0]
                x_num_next = x_num_next.detach() - harpoon_eta * grad.detach()
                corrections += int(valid.sum().item())

        x_num = x_num_next.detach()
        x_cat = x_cat_next.detach()

    synchronize(device)
    elapsed = time.perf_counter() - start
    final_encoded = torch.cat([x_num, x_cat.float()], dim=1).cpu()
    final_df = decode_tensor(bundle, final_encoded)
    if collect_trace:
        for row in trace:
            preview = row.pop("preview_encoded")
            row["numeric_mae_to_final_encoded"] = float((preview[:, : diffusion.num_numerical_features] - final_encoded[:, : diffusion.num_numerical_features]).abs().mean().item())
            row["categorical_agreement_to_final"] = float((preview[:, diffusion.num_numerical_features :].long() == final_encoded[:, diffusion.num_numerical_features :].long()).float().mean().item())
    return final_df, final_encoded, pd.DataFrame(trace), elapsed, corrections, feedback_cases


def forward_probe(bundle: ModelBundle, n: int, repeats: int) -> pd.DataFrame:
    diffusion = bundle.diffusion
    x0 = bundle.dataset_test.X[:n].to(bundle.device)
    x0_num = x0[:, : diffusion.num_numerical_features]
    x0_cat = x0[:, diffusion.num_numerical_features :].long()
    times = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.03, 0.0]
    rows = []
    for repeat in range(repeats):
        seed_everything(1000 + repeat)
        epsilon = torch.randn_like(x0_num)
        uniforms = torch.rand_like(x0_cat.float())
        for t_value in times:
            t = torch.tensor([[t_value]], device=bundle.device)
            sigma_num = diffusion.num_schedule.total_noise(t).squeeze(0)
            sigma_cat = diffusion.cat_schedule.total_noise(t).squeeze(0)
            x_num_t = x0_num + epsilon * sigma_num
            move = -torch.expm1(-sigma_cat)
            x_cat_t = torch.where(uniforms < move[None, :], diffusion.mask_index[None, :], x0_cat)
            with torch.no_grad():
                _, denoised, raw_logits, cat_pred = model_preview(bundle, x_num_t, x_cat_t, t, sigma_num)
            decoded = decode_tensor(bundle, torch.cat([denoised, cat_pred.float()], dim=1))
            flags = rule_flags(decoded, forward_probe.rule_context)
            rows.append(
                {
                    "repeat": repeat,
                    "t": t_value,
                    "sigma_num_mean": float(sigma_num.mean().item()),
                    "numeric_mae_encoded": float((denoised - x0_num).abs().mean().item()),
                    "numeric_rmse_encoded": float(torch.sqrt((denoised - x0_num).square().mean()).item()),
                    "categorical_accuracy": float((cat_pred == x0_cat).float().mean().item()),
                    "categorical_mask_fraction": float((x_cat_t == diffusion.mask_index[None, :]).float().mean().item()),
                    "hard_rule_violation_rate": float(flags["any_hard"].mean()),
                }
            )
    return pd.DataFrame(rows)


def post_repair(df: pd.DataFrame, context: RuleContext) -> pd.DataFrame:
    result = clean_strings(df)
    expected = result["education"].map(context.education_map)
    result.loc[expected.notna(), "education.num"] = expected[expected.notna()].astype(float)
    husband = result["relationship"].eq("Husband")
    wife = result["relationship"].eq("Wife")
    result.loc[husband, "sex"] = "Male"
    result.loc[wife, "sex"] = "Female"
    result.loc[husband | wife, "marital.status"] = "Married-civ-spouse"
    result["capital.gain"] = pd.to_numeric(result["capital.gain"], errors="coerce").clip(lower=0)
    result["capital.loss"] = pd.to_numeric(result["capital.loss"], errors="coerce").clip(lower=0)
    result["hours.per.week"] = pd.to_numeric(result["hours.per.week"], errors="coerce").clip(lower=1)
    return result


def choose_best_of(candidates: list[pd.DataFrame], context: RuleContext) -> pd.DataFrame:
    scores = np.stack([rule_flags(df, context).drop(columns="any_hard").sum(axis=1).to_numpy() for df in candidates], axis=1)
    chosen = scores.argmin(axis=1)
    arrays = [df.reset_index(drop=True) for df in candidates]
    rows = [arrays[int(choice)].iloc[i] for i, choice in enumerate(chosen)]
    return pd.DataFrame(rows).reset_index(drop=True)


def marginal_metrics(real: pd.DataFrame, syn: pd.DataFrame, num_cols: list[str], cat_cols: list[str]):
    ks = []
    for col in num_cols:
        r = pd.to_numeric(real[col], errors="coerce").dropna()
        s = pd.to_numeric(syn[col], errors="coerce").dropna()
        if len(r) and len(s):
            ks.append(ks_2samp(r, s).statistic)
    tv = []
    for col in cat_cols:
        rp = clean_strings(real[[col]])[col].value_counts(normalize=True)
        sp = clean_strings(syn[[col]])[col].value_counts(normalize=True)
        idx = rp.index.union(sp.index)
        tv.append(0.5 * float((rp.reindex(idx, fill_value=0) - sp.reindex(idx, fill_value=0)).abs().sum()))
    return float(np.mean(ks)), float(np.mean(tv))


def joint_corr_mae(real: pd.DataFrame, syn: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> float:
    real = clean_strings(real)
    syn = clean_strings(syn)
    transformer = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    x_real = np.asarray(transformer.fit_transform(real))
    x_syn = np.asarray(transformer.transform(syn))
    corr_real = np.nan_to_num(np.corrcoef(x_real, rowvar=False))
    corr_syn = np.nan_to_num(np.corrcoef(x_syn, rowvar=False))
    upper = np.triu_indices_from(corr_real, k=1)
    return float(np.mean(np.abs(corr_real[upper] - corr_syn[upper])))


def rare_coverage(real: pd.DataFrame, syn: pd.DataFrame, cat_cols: list[str]) -> tuple[float, float]:
    real = clean_strings(real)
    syn = clean_strings(syn)
    covered, total = 0, 0
    threshold = max(20, int(0.005 * len(real)))
    for col in cat_cols:
        counts = real[col].value_counts()
        rare = set(counts[(counts >= 2) & (counts <= threshold)].index)
        total += len(rare)
        covered += len(rare & set(syn[col].unique()))
    pair_cols = [("relationship", "sex"), ("education", "income"), ("race", "income")]
    pair_covered, pair_total = 0, 0
    for a, b in pair_cols:
        counts = real.groupby([a, b]).size()
        rare = set(counts[(counts >= 2) & (counts <= threshold)].index)
        present = set(map(tuple, syn[[a, b]].to_numpy()))
        pair_total += len(rare)
        pair_covered += len(rare & present)
    return (covered / total if total else float("nan"), pair_covered / pair_total if pair_total else float("nan"))


def conditional_income_gap(real: pd.DataFrame, syn: pd.DataFrame) -> float:
    real, syn = clean_strings(real), clean_strings(syn)
    gaps = []
    for group_col in ("sex", "race", "education"):
        real_rate = real.assign(y=real["income"].eq(">50K")).groupby(group_col)["y"].agg(["mean", "size"])
        syn_rate = syn.assign(y=syn["income"].eq(">50K")).groupby(group_col)["y"].mean()
        valid = real_rate[real_rate["size"] >= 30].index.intersection(syn_rate.index)
        gaps.extend((real_rate.loc[valid, "mean"] - syn_rate.loc[valid]).abs().tolist())
    return float(np.mean(gaps)) if gaps else float("nan")


def utility_metrics(syn: pd.DataFrame, test: pd.DataFrame, target: str, num_cols: list[str], cat_cols: list[str]):
    train = clean_strings(syn)
    test = clean_strings(test)
    features_num = [c for c in num_cols if c != target]
    features_cat = [c for c in cat_cols if c != target]
    processor = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
        ]
    )
    model = Pipeline([("prep", processor), ("clf", LogisticRegression(max_iter=400, class_weight="balanced"))])
    y_train = train[target].eq(">50K").astype(int)
    y_test = test[target].eq(">50K").astype(int)
    if y_train.nunique() < 2:
        return float("nan"), float("nan")
    model.fit(train[features_num + features_cat], y_train)
    prob = model.predict_proba(test[features_num + features_cat])[:, 1]
    pred = prob >= 0.5
    return float(roc_auc_score(y_test, prob)), float(f1_score(y_test, pred))


def evaluate_variant(name: str, syn: pd.DataFrame, real: pd.DataFrame, test: pd.DataFrame, context: RuleContext, bundle: ModelBundle, runtime: float, denoiser_passes: int, corrections: int):
    num_cols = bundle.num_names
    cat_cols = bundle.cat_names
    flags = rule_flags(syn, context)
    ks, tv = marginal_metrics(real, syn, num_cols, cat_cols)
    rare_value, rare_pair = rare_coverage(real, syn, cat_cols)
    auroc, f1 = utility_metrics(syn, test, "income", num_cols, cat_cols)
    return {
        "variant": name,
        "samples": len(syn),
        "hard_any_violation_rate": float(flags["any_hard"].mean()),
        **{f"violation_{c}": float(flags[c].mean()) for c in flags.columns if c != "any_hard"},
        "numeric_ks_mean": ks,
        "categorical_tvd_mean": tv,
        "joint_corr_mae_proxy": joint_corr_mae(real, syn, num_cols, cat_cols),
        "conditional_income_gap": conditional_income_gap(real, syn),
        "rare_value_coverage": rare_value,
        "rare_pair_coverage": rare_pair,
        "utility_auroc": auroc,
        "utility_f1": f1,
        "duplicate_rate": float(syn.astype(str).duplicated().mean()),
        "runtime_seconds": runtime,
        "denoiser_forward_passes": denoiser_passes,
        "llm_calls": 0,
        "typed_numeric_corrections": corrections,
    }


def plot_trajectories(forward: pd.DataFrame, free: pd.DataFrame, output: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot trajectory diagnostics")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    agg = forward.groupby("t").mean(numeric_only=True).sort_index()
    axes[0].plot(agg.index, agg["numeric_mae_encoded"], marker="o")
    axes[0].set(xlabel="forward-noise t", ylabel="MAE (encoded)", title="Known-x0 numeric readability")
    axes[1].plot(agg.index, agg["categorical_accuracy"], marker="o")
    axes[1].set(xlabel="forward-noise t", ylabel="accuracy", title="Known-x0 categorical readability", ylim=(0, 1.02))
    free_sorted = free.sort_values("t")
    axes[2].plot(free_sorted["t"], free_sorted["categorical_agreement_to_final"], marker="o", label="cat agreement")
    axes[2].plot(free_sorted["t"], 1 - free_sorted["preview_rule_violation_rate"], marker="s", label="rule-valid fraction")
    axes[2].set(xlabel="reverse trajectory t", ylabel="fraction", title="Free-generation preview stability", ylim=(0, 1.02))
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--probe-rows", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--intervention-t", type=float, default=0.35)
    parser.add_argument("--harpoon-eta", type=float, default=0.03)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    seed_everything(args.seed)
    bundle = build_model(args.repo, args.checkpoint, args.config, args.steps, device)
    real = pd.read_csv(args.repo / "synthetic/adult/real.csv")
    test = pd.read_csv(args.repo / "synthetic/adult/test.csv")
    context = make_rule_context(real)
    forward_probe.rule_context = context
    sample_variant.rule_context = context

    forward = forward_probe(bundle, min(args.probe_rows, len(bundle.dataset_test)), repeats=3)
    forward.to_csv(args.output / "trajectory_forward_known_x0.csv", index=False)
    forward.groupby("t").agg(["mean", "std"]).to_csv(args.output / "trajectory_forward_known_x0_summary.csv")

    variants: dict[str, tuple[pd.DataFrame, float, int, int]] = {}
    no_df, _, free_trace, no_time, no_corr, feedback_cases = sample_variant(
        bundle, args.samples, args.steps, args.seed, "no_guidance", args.intervention_t, args.harpoon_eta, collect_trace=True
    )
    free_trace.to_csv(args.output / "trajectory_free_generation.csv", index=False)
    with (args.output / "feedback_cases.jsonl").open("w") as handle:
        for case in feedback_cases:
            handle.write(json.dumps(case) + "\n")
    variants["no_guidance"] = (no_df, no_time, args.steps, no_corr)

    fixed_df, _, _, fixed_time, fixed_corr, _ = sample_variant(
        bundle, args.samples, args.steps, args.seed, "fixed_rule_denoising_proxy", args.intervention_t, args.harpoon_eta
    )
    variants["fixed_rule_denoising_proxy"] = (fixed_df, fixed_time, args.steps, fixed_corr)
    dynamic_df, _, _, dynamic_time, dynamic_corr, _ = sample_variant(
        bundle, args.samples, args.steps, args.seed, "dynamic_record_feedback_proxy", args.intervention_t, args.harpoon_eta
    )
    variants["dynamic_record_feedback_proxy"] = (dynamic_df, dynamic_time, args.steps, dynamic_corr)
    late_dynamic_df, _, _, late_dynamic_time, late_dynamic_corr, _ = sample_variant(
        bundle, args.samples, args.steps, args.seed, "dynamic_record_feedback_proxy", 0.10, args.harpoon_eta
    )
    variants["dynamic_record_feedback_late_t0.10_proxy"] = (
        late_dynamic_df,
        late_dynamic_time,
        args.steps,
        late_dynamic_corr,
    )
    harpoon_df, _, _, harpoon_time, harpoon_corr, _ = sample_variant(
        bundle, args.samples, args.steps, args.seed, "harpoon_numeric_proxy", args.intervention_t, args.harpoon_eta
    )
    variants["harpoon_numeric_proxy"] = (harpoon_df, harpoon_time, args.steps, harpoon_corr)
    variants["post_repair_proxy"] = (post_repair(no_df, context), no_time, args.steps, int(rule_flags(no_df, context)["any_hard"].sum()))

    bon_candidates = [no_df]
    bon_runtime = no_time
    for branch in range(1, 4):
        candidate, _, _, branch_time, _, _ = sample_variant(
            bundle, args.samples, args.steps, args.seed + branch, "no_guidance", args.intervention_t, args.harpoon_eta
        )
        bon_candidates.append(candidate)
        bon_runtime += branch_time
    variants["best_of_4_verifier"] = (choose_best_of(bon_candidates, context), bon_runtime, 4 * args.steps, 0)

    metrics = []
    for name, (frame, runtime, passes, corrections) in variants.items():
        frame.to_csv(args.output / f"samples_{name}.csv", index=False)
        metrics.append(evaluate_variant(name, frame, real, test, context, bundle, runtime, passes, corrections))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.output / "intervention_metrics.csv", index=False)
    plot_trajectories(forward, free_trace, args.output / "trajectory_readability.png")

    metadata = {
        "status": "executed",
        "scope": "mechanism pilot; proxy baselines, not paper reproductions",
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "checkpoint_training_steps": pickle.load(args.config.open("rb"))["train"]["main"]["steps"],
        "device": str(device),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "samples": args.samples,
        "probe_rows": args.probe_rows,
        "sampler_steps": args.steps,
        "intervention_t": args.intervention_t,
        "harpoon_eta": args.harpoon_eta,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
