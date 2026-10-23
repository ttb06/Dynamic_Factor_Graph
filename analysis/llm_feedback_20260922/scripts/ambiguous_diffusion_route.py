#!/usr/bin/env python3
"""Test row-specific graph-edge/actuator selection with TabDiff imputation.

The observed violation is identical (count=0, duration>0), while controlled
corruption determines whether count or duration is the damaged endpoint.
This isolates routing from constraint parsing and from final certification.
"""
from __future__ import annotations

import argparse, copy, json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ambiguous_repair_audit import PAIRS, fit_router, make_cases, model_features
from conditioned_tabdiff_adapter import decode, seed_all
from tabdiff_probe import preprocess_dataset


def context_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Common context that cannot reveal the edited count-duration endpoint."""
    excluded = {"Revenue", "_route", "_clean_count", "_clean_duration"}
    excluded |= {name for pair in PAIRS for name in pair}
    columns = [c for c in frame.columns if c not in excluded]
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    return numeric, [c for c in columns if c not in numeric]


def fit_common_router(frame: pd.DataFrame) -> Pipeline:
    numeric, categorical = context_columns(frame)
    transform = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("features", transform),
                      ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    model.fit(frame[numeric + categorical], frame["_route"])
    return model


def natural_state_training(train: pd.DataFrame, count: str, duration: str,
                           rng: np.random.Generator, limit: int) -> pd.DataFrame:
    zero = train[(train[count] == 0) & (train[duration] == 0)].copy()
    positive = train[(train[count] > 0) & (train[duration] > 0)].copy()
    n = min(len(zero), len(positive), limit)
    zero = zero.iloc[rng.choice(len(zero), n, replace=False)]; zero["_route"] = 0
    positive = positive.iloc[rng.choice(len(positive), n, replace=False)]; positive["_route"] = 1
    return pd.concat([zero, positive], ignore_index=True)


def encode_frame(frame: pd.DataFrame, dataset, info: dict) -> torch.Tensor:
    num_names = [info["column_names"][i] for i in info["num_col_idx"]]
    cat_names = [info["column_names"][i] for i in info["cat_col_idx"]]
    target_names = [info["column_names"][i] for i in info["target_col_idx"]]
    num = dataset.num_transform.transform(frame[num_names].to_numpy(float))
    # TabDiff preprocessing places the target before ordinary categoricals.
    cat = dataset.cat_transform.transform(frame[target_names + cat_names].astype(str).to_numpy())
    return torch.tensor(np.concatenate([num, cat], axis=1), dtype=torch.float32)


def certify(frame: pd.DataFrame, route: np.ndarray, count: str, duration: str) -> pd.DataFrame:
    out = frame.copy()
    bad = pd.to_numeric(out[count], errors="coerce").lt(.5) & pd.to_numeric(out[duration], errors="coerce").gt(1e-8)
    out.loc[bad & (route == 0), duration] = 0.0
    out.loc[bad & (route == 1), count] = 1.0
    return out


@torch.no_grad()
def routed_impute(diffusion, dataset, info, observed: pd.DataFrame, route: np.ndarray,
                  count: str, duration: str, rounds: int, seed: int) -> pd.DataFrame:
    encoded = encode_frame(observed, dataset, info)
    result = observed[info["column_names"]].copy()
    num_names = [info["column_names"][i] for i in info["num_col_idx"]]
    for selected_route, column in ((0, duration), (1, count)):
        index = np.flatnonzero(route == selected_route)
        if not len(index):
            continue
        subset = encoded[index]
        seed_all(seed + selected_route)
        sampled = diffusion.sample_impute(
            subset[:, :dataset.d_numerical].to(diffusion.device),
            subset[:, dataset.d_numerical:].long().to(diffusion.device),
            [num_names.index(column)], [], rounds, "x_t", 0.0, 0.0,
        )
        decoded = decode(dataset, sampled, info)
        result.loc[index, column] = pd.to_numeric(decoded[column], errors="coerce").to_numpy(float)
    return result


def measure(frame: pd.DataFrame, cases: pd.DataFrame, route: np.ndarray, truth: np.ndarray,
            train: pd.DataFrame, count: str, duration: str, method: str, certified: bool) -> dict:
    pc = pd.to_numeric(frame[count], errors="coerce").to_numpy(float)
    pdur = pd.to_numeric(frame[duration], errors="coerce").to_numpy(float)
    cs, ds = max(float(train[count].std()), 1e-8), max(float(train[duration].std()), 1e-8)
    error = np.abs(pc - cases["_clean_count"].to_numpy(float)) / cs + np.abs(pdur - cases["_clean_duration"].to_numpy(float)) / ds
    valid = ~((pc < .5) & (pdur > 1e-8))
    return {"relation": f"{count}=0 -> {duration}=0", "method": method, "certified": certified,
            "cases": len(cases), "route_accuracy": float(np.mean(route == truth)),
            "validity": float(valid.mean()), "pair_error": float(error.mean()),
            "p90_pair_error": float(np.quantile(error, .9)), "wrong_column_edit": float(np.mean(route != truth))}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--steps", type=int, default=30); p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-train-per-route", type=int, default=2500); p.add_argument("--max-test-per-route", type=int, default=400)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    started = time.perf_counter(); sys.path.insert(0, str(a.repo)); device = torch.device("cuda")
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP
    info = json.loads((a.repo / "data/shoppers/info.json").read_text())
    dataset, _ = preprocess_dataset(a.repo, "shoppers")
    with a.config.open("rb") as f: cfg = pickle.load(f)
    params = copy.deepcopy(cfg["unimodmlp_params"]); params["d_numerical"] = dataset.d_numerical; params["categories"] = (dataset.categories + 1).tolist()
    model = Model(UniModMLP(**params), **cfg["diffusion_params"]["edm_params"])
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False); model.load_state_dict(state["denoise_fn"])
    dp = copy.deepcopy(cfg["diffusion_params"]); dp.update(num_timesteps=a.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    dp["sampler_params"] = copy.deepcopy(dp["sampler_params"]); dp["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(dataset.categories, dataset.d_numerical, model, None, **dp, device=device).to(device)
    diffusion.num_schedule.load_state_dict(state["num_schedule"]); diffusion.cat_schedule.load_state_dict(state["cat_schedule"]); diffusion.eval()
    train = pd.read_csv(a.repo / "data/shoppers/train.csv"); test = pd.read_csv(a.repo / "data/shoppers/test.csv")
    rng = np.random.default_rng(a.seed); rows = []
    # A leave-one-relation-out router uses controlled cases from other edges
    # and only common context features, never the tested endpoints.
    cross_training = {}
    for held_count, held_duration in PAIRS:
        parts = []
        for other_count, other_duration in PAIRS:
            if other_count == held_count: continue
            try: parts.append(make_cases(train, other_count, other_duration, rng, a.max_train_per_route))
            except RuntimeError: pass
        if parts: cross_training[held_count] = fit_common_router(pd.concat(parts, ignore_index=True))
    for ordinal, (count, duration) in enumerate(PAIRS):
        try:
            train_cases = make_cases(train, count, duration, rng, a.max_train_per_route)
            cases = make_cases(test, count, duration, rng, a.max_test_per_route)
        except RuntimeError:
            continue
        router = fit_router(train_cases, count); x, _, _ = model_features(cases, count)
        truth = cases["_route"].to_numpy(int)
        numeric_ctx, categorical_ctx = context_columns(cases)
        natural = fit_common_router(natural_state_training(train, count, duration, rng, a.max_train_per_route))
        routes = {"fixed_duration": np.zeros(len(cases), int), "fixed_count": np.ones(len(cases), int),
                  "context_router": router.predict(x).astype(int),
                  "natural_state_router": natural.predict(cases[numeric_ctx + categorical_ctx]).astype(int),
                  "oracle_route": truth}
        if count in cross_training:
            routes["cross_relation_router"] = cross_training[count].predict(
                cases[numeric_ctx + categorical_ctx]).astype(int)
        observed = cases[info["column_names"]]
        for name, route in routes.items():
            raw = routed_impute(diffusion, dataset, info, observed, route, count, duration, a.rounds,
                                a.seed + ordinal * 1000)
            rows.append(measure(raw, cases, route, truth, train, count, duration, name, False))
            rows.append(measure(certify(raw, route, count, duration), cases, route, truth, train,
                                count, duration, name, True))
    metrics = pd.DataFrame(rows); metrics.to_csv(a.output / "metrics.csv", index=False)
    summary = metrics.groupby(["method", "certified"]).agg(relations=("relation", "nunique"), route_accuracy=("route_accuracy", "mean"),
        validity=("validity", "mean"), pair_error=("pair_error", "mean"), p90_error=("p90_pair_error", "mean")).reset_index()
    summary.to_csv(a.output / "summary.csv", index=False)
    lines = ["# Row-specific diffusion actuation", "", "| Method | Certified | Route accuracy | Validity | Pair error | P90 error |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary.itertuples(index=False): lines.append(f"| {r.method} | {r.certified} | {r.route_accuracy:.4f} | {r.validity:.4f} | {r.pair_error:.4f} | {r.p90_error:.4f} |")
    (a.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (a.output / "run_metadata.json").write_text(json.dumps({"status":"executed", "scope":"controlled corruption; routing oracle used only as upper bound", "seed":a.seed, "steps":a.steps, "rounds":a.rounds, "runtime_seconds":time.perf_counter()-started}, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__": main()
