#!/usr/bin/env python3
"""Compare paired intervention contrasts across two completed GPU replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTRASTS = (
    ("late_once", "no_guidance"),
    ("persistent_late_once", "late_once"),
    ("late_recheck_flagged", "late_once"),
    ("late_recheck_all", "late_recheck_flagged"),
    ("post_repair", "late_recheck_all"),
)
METRICS = (
    "hard_any_violation_rate",
    "numeric_ks_mean",
    "categorical_tvd_mean",
    "joint_corr_mae_proxy",
    "conditional_income_gap",
    "rare_pair_coverage",
    "utility_auroc",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.replay) != len(args.label):
        raise ValueError("--replay and --label counts must match")
    args.output.mkdir(parents=True, exist_ok=False)

    blocks = []
    for label, directory in zip(args.label, args.replay):
        frame = pd.read_csv(directory / "intervention_metrics_by_seed.csv")
        frame["replay"] = label
        blocks.append(frame)
    all_rows = pd.concat(blocks, ignore_index=True)
    all_rows.to_csv(args.output / "combined_replays.csv", index=False)

    contrast_rows = []
    for replay, replay_frame in all_rows.groupby("replay"):
        for seed, seed_frame in replay_frame.groupby("seed"):
            indexed = seed_frame.set_index("variant")
            for treatment, reference in CONTRASTS:
                for metric in METRICS:
                    contrast_rows.append(
                        {
                            "replay": replay,
                            "seed": seed,
                            "treatment": treatment,
                            "reference": reference,
                            "metric": metric,
                            "difference_treatment_minus_reference": indexed.loc[treatment, metric]
                            - indexed.loc[reference, metric],
                        }
                    )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.output / "paired_contrasts.csv", index=False)
    grouped = contrasts.groupby(["treatment", "reference", "metric"])[
        "difference_treatment_minus_reference"
    ]
    summary = grouped.agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary["fraction_negative"] = grouped.apply(lambda values: float((values < 0).mean())).to_numpy()
    summary["fraction_positive"] = grouped.apply(lambda values: float((values > 0).mean())).to_numpy()
    summary.to_csv(args.output / "contrast_stability_summary.csv", index=False)

    shared = all_rows.pivot_table(index=["seed", "variant"], columns="replay", values=list(METRICS))
    replay_labels = args.label[:2]
    replay_differences = []
    if len(replay_labels) >= 2:
        for metric in METRICS:
            values = (shared[(metric, replay_labels[1])] - shared[(metric, replay_labels[0])]).abs()
            replay_differences.append(
                {
                    "metric": metric,
                    "mean_absolute_replay_difference": float(values.mean()),
                    "max_absolute_replay_difference": float(values.max()),
                }
            )
    pd.DataFrame(replay_differences).to_csv(args.output / "absolute_replay_differences.csv", index=False)
    metadata = {
        "status": "executed",
        "scope": "execution replay stability, not independent checkpoints",
        "replays": {label: str(path) for label, path in zip(args.label, args.replay)},
        "note": "Sign consistency is used for planning; the six blocks are not treated as six independent model fits.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    focus = summary[
        summary["metric"].isin(["hard_any_violation_rate", "rare_pair_coverage"])
    ]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
