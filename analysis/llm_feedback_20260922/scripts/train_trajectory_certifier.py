#!/usr/bin/env python3
"""Train per-edge final-violation certifiers from synthetic diffusion trajectories."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-class", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    data = np.load(args.input)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.uint8)
    edge_ids = data["edge_ids"].astype(str)
    sample_ids = data["sample_ids"].astype(np.int64)
    validation = sample_ids % 5 == 0
    rows, models = [], {}

    for edge_id in sorted(np.unique(edge_ids)):
        edge = edge_ids == edge_id
        train = edge & ~validation
        valid = edge & validation
        train_pos = int(labels[train].sum())
        train_neg = int(train.sum() - train_pos)
        valid_pos = int(labels[valid].sum())
        valid_neg = int(valid.sum() - valid_pos)
        if min(train_pos, train_neg, valid_pos, valid_neg) < args.min_class:
            rows.append({"candidate_id": edge_id, "train_rows": int(train.sum()),
                "valid_rows": int(valid.sum()), "train_positive": train_pos,
                "valid_positive": valid_pos, "status": "insufficient_class_support"})
            continue

        model = HistGradientBoostingClassifier(max_iter=args.max_iter, learning_rate=.06,
            max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0,
            random_state=args.seed)
        model.fit(features[train], labels[train])
        probability = model.predict_proba(features[valid])[:, 1]
        simple = LogisticRegression(max_iter=500, class_weight="balanced", random_state=args.seed)
        simple.fit(features[train, -2:], labels[train])
        simple_probability = simple.predict_proba(features[valid, -2:])[:, 1]
        rows.append({"candidate_id": edge_id, "train_rows": int(train.sum()),
            "valid_rows": int(valid.sum()), "train_positive": train_pos,
            "valid_positive": valid_pos, "status": "trained",
            "rich_auroc": float(roc_auc_score(labels[valid], probability)),
            "rich_average_precision": float(average_precision_score(labels[valid], probability)),
            "rich_brier": float(brier_score_loss(labels[valid], probability)),
            "time_streak_auroc": float(roc_auc_score(labels[valid], simple_probability)),
            "time_streak_average_precision": float(average_precision_score(labels[valid], simple_probability)),
            "prevalence": float(labels[valid].mean())})
        models[edge_id] = model

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "certifier_metrics.csv", index=False)
    trained = metrics[metrics.status.eq("trained")]
    summary = {
        "input": str(args.input), "rows": int(len(features)),
        "features": int(features.shape[1]), "trained_edges": int(len(models)),
        "mean_rich_auroc": float(trained.rich_auroc.mean()) if len(trained) else None,
        "mean_time_streak_auroc": float(trained.time_streak_auroc.mean()) if len(trained) else None,
        "mean_rich_average_precision": float(trained.rich_average_precision.mean()) if len(trained) else None,
        "mean_prevalence": float(trained.prevalence.mean()) if len(trained) else None,
        "split": "sample_id modulo 5; all timesteps for one trajectory remain in one split",
    }
    with (args.output / "trajectory_certifier.pkl").open("wb") as handle:
        pickle.dump({"models": models, "n_features": int(features.shape[1]),
                     "feature_layout": "encoded predicted-clean state + t + log1p(streak)"}, handle)
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Trajectory certifier validation", "",
        f"Rows: {summary['rows']}; trained edges: {summary['trained_edges']}; "
        f"mean rich AUROC: {summary['mean_rich_auroc']}; "
        f"mean time+streak AUROC: {summary['mean_time_streak_auroc']}.", "",
        "| Edge | Validation rows | Positives | Prevalence | Rich AUROC | Time+streak AUROC | Rich AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in trained.itertuples(index=False):
        lines.append(f"| {row.candidate_id} | {row.valid_rows} | {row.valid_positive} | "
            f"{row.prevalence:.4f} | {row.rich_auroc:.4f} | {row.time_streak_auroc:.4f} | "
            f"{row.rich_average_precision:.4f} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
