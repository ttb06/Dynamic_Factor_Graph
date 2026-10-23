#!/usr/bin/env python3
"""Independent, lightweight audit of an official LCATS run.

The audit deliberately uses metrics outside LCATS' candidate-selection score.  It
compares the original TabSyn sample and every aligned candidate against a held-out
real split, and reports validity, fidelity, downstream utility, and repair cost.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_verifier(path: Path):
    spec = importlib.util.spec_from_file_location("lcats_verify_independent", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.verify


def schema(reference: pd.DataFrame):
    numeric = [c for c in reference.columns if pd.api.types.is_numeric_dtype(reference[c])]
    categorical = [c for c in reference.columns if c not in numeric]
    return numeric, categorical


def marginal_metrics(reference: pd.DataFrame, sample: pd.DataFrame, numeric, categorical):
    ks = {c: float(ks_2samp(reference[c].dropna(), sample[c].dropna()).statistic) for c in numeric}
    tv = {}
    for c in categorical:
        levels = sorted(set(reference[c].astype(str)) | set(sample[c].astype(str)))
        p = reference[c].astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0)
        q = sample[c].astype(str).value_counts(normalize=True).reindex(levels, fill_value=0.0)
        tv[c] = float(0.5 * np.abs(p - q).sum())
    corr_ref = reference[numeric].corr(method="spearman").fillna(0.0).to_numpy()
    corr_syn = sample[numeric].corr(method="spearman").fillna(0.0).to_numpy()
    upper = np.triu_indices_from(corr_ref, k=1)
    corr_mae = float(np.abs(corr_ref[upper] - corr_syn[upper]).mean()) if len(upper[0]) else 0.0
    return {
        "mean_numeric_ks": float(np.mean(list(ks.values()))) if ks else 0.0,
        "max_numeric_ks": float(np.max(list(ks.values()))) if ks else 0.0,
        "mean_categorical_tv": float(np.mean(list(tv.values()))) if tv else 0.0,
        "max_categorical_tv": float(np.max(list(tv.values()))) if tv else 0.0,
        "spearman_corr_mae": corr_mae,
        "numeric_ks_by_column": ks,
        "categorical_tv_by_column": tv,
    }


def propensity_auc(reference: pd.DataFrame, sample: pd.DataFrame, numeric, categorical, seed: int):
    n = min(len(reference), len(sample))
    real = reference.sample(n=n, random_state=seed).copy()
    syn = sample.sample(n=n, random_state=seed).copy()
    x = pd.concat([real, syn], ignore_index=True)
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=2000, random_state=seed))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prob = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, prob))
    return max(auc, 1.0 - auc)


def tstr_regression(train: pd.DataFrame, test: pd.DataFrame, target: str, numeric, categorical, seed: int):
    features = [c for c in train.columns if c != target]
    nums = [c for c in numeric if c in features]
    cats = [c for c in categorical if c in features]
    pre = ColumnTransformer([
        ("num", "passthrough", nums),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ])
    model = make_pipeline(
        pre,
        RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
    )
    model.fit(train[features], train[target])
    pred = model.predict(test[features])
    rmse = float(mean_squared_error(test[target], pred) ** 0.5)
    scale = float(test[target].std(ddof=0)) or 1.0
    return {"tstr_rmse": rmse, "tstr_nrmse": rmse / scale, "tstr_r2": float(r2_score(test[target], pred))}


def validity(verify, frame: pd.DataFrame):
    masks = [pd.Series(m, index=frame.index).astype(bool) for m in verify(frame)]
    matrix = np.column_stack([m.to_numpy() for m in masks])
    return {
        "row_any_violation_rate": float(matrix.any(axis=1).mean()),
        "cell_constraint_violation_rate": float(matrix.mean()),
        "violations_by_constraint": [float(matrix[:, i].mean()) for i in range(matrix.shape[1])],
    }


def repair_cost(before: pd.DataFrame, after: pd.DataFrame, numeric, categorical):
    if len(before) != len(after):
        return {"row_aligned": False, "before_rows": len(before), "after_rows": len(after)}
    b, a = before.reset_index(drop=True), after.reset_index(drop=True)
    changes = pd.DataFrame(False, index=b.index, columns=b.columns)
    numeric_l1 = []
    for c in numeric:
        delta = (a[c] - b[c]).abs()
        changes[c] = delta > 1e-9
        scale = float(b[c].quantile(0.75) - b[c].quantile(0.25)) or float(b[c].std(ddof=0)) or 1.0
        numeric_l1.append(float((delta / scale).mean()))
    for c in categorical:
        changes[c] = a[c].astype(str) != b[c].astype(str)
    return {
        "row_aligned": True,
        "row_edit_rate": float(changes.any(axis=1).mean()),
        "cell_edit_rate": float(changes.to_numpy().mean()),
        "mean_numeric_normalized_l1": float(np.mean(numeric_l1)) if numeric_l1 else 0.0,
        "categorical_cell_edit_rate": float(changes[categorical].to_numpy().mean()) if categorical else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lcats-root", type=Path, required=True)
    p.add_argument("--dataset", default="insurance")
    p.add_argument("--target", default="charges")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261002)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    data_dir = args.lcats_root / "data" / args.dataset
    syn_dir = args.lcats_root / "synthetic" / args.dataset
    real_train = pd.read_csv(syn_dir / "real.csv")
    real_test = pd.read_csv(syn_dir / "test.csv")
    tabsyn = pd.read_csv(syn_dir / "tabsyn.csv")
    verify = load_verifier(data_dir / "verify_script.py")
    numeric, categorical = schema(real_train)

    candidates = {"real_train_ceiling": real_train, "tabsyn_original": tabsyn}
    for path in sorted(syn_dir.glob("aligned_data_trial_*.csv")):
        candidates[path.stem] = pd.read_csv(path)
    if (syn_dir / "aligned_data.csv").exists():
        candidates["aligned_selected"] = pd.read_csv(syn_dir / "aligned_data.csv")

    rows, details = [], {}
    for name, frame in candidates.items():
        metrics = {}
        metrics.update(validity(verify, frame))
        metrics.update(marginal_metrics(real_train, frame, numeric, categorical))
        metrics["propensity_auc"] = propensity_auc(real_train, frame, numeric, categorical, args.seed)
        metrics.update(tstr_regression(frame, real_test, args.target, numeric, categorical, args.seed))
        if name.startswith("aligned"):
            metrics.update(repair_cost(tabsyn, frame, numeric, categorical))
        details[name] = metrics
        rows.append({k: v for k, v in {"candidate": name, **metrics}.items() if not isinstance(v, (dict, list))})

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "candidate_metrics.csv", index=False)
    (args.output / "details.json").write_text(json.dumps(details, indent=2))

    cols = [
        "candidate", "row_any_violation_rate", "mean_numeric_ks", "mean_categorical_tv",
        "spearman_corr_mae", "propensity_auc", "tstr_nrmse", "tstr_r2", "row_edit_rate", "cell_edit_rate",
    ]
    lines = [
        "# Independent audit of official LCATS outputs",
        "",
        "Lower is better except `tstr_r2`. Propensity AUC is ideal at 0.5. The evaluator is not used by LCATS to select candidates.",
        "",
        "| Candidate | Any violation | Numeric KS | Categorical TV | Corr. MAE | Propensity AUC | TSTR NRMSE | TSTR R2 | Row edit | Cell edit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in table.reindex(columns=cols).iterrows():
        fmt = lambda x: "—" if pd.isna(x) else f"{float(x):.4f}"
        lines.append("| " + " | ".join([str(r["candidate"])] + [fmt(r[c]) for c in cols[1:]]) + " |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(table.to_string(index=False))
    print(args.output / "RESULTS.md")


if __name__ == "__main__":
    main()
