#!/usr/bin/env python3
"""Aggregate the focused three-seed subgroup lookahead confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = [
    "lookahead_anchored_violated_numgraph_t100_plus_certifier",
    "lookahead_anchored_violated_topk3_t100_plus_certifier",
    "lookahead_anchored_violated_topk5_t100_plus_certifier",
    "lookahead_anchored_violated_masked_t100_plus_certifier",
    "lookahead_anchored_violated_t100_plus_certifier",
    "empirical_marginal_bad_postrepair",
    "diffusion_impute_then_certify",
    "diffusion_impute_numgraph_then_certify",
    "diffusion_bestof8_target_then_certify",
    "diffusion_bestof8_numgraph_then_certify",
    "subgroup_final_repair",
]
METRICS = [
    "active_satisfaction", "active_drift_sd", "active_changed",
    "active_target_ks_reference", "active_target_wasserstein_sd",
    "active_cohort_numeric_ks", "active_cohort_categorical_tv",
    "target_context_spearman_mae", "exception_changed", "antecedent_flip", "subgroup_flip",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    frames = []
    for run in args.runs:
        path = run / "policy_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        metadata = json.loads((run / "run_metadata.json").read_text())
        frame["sampling_seed"] = int(metadata["seed"])
        frame["run"] = run.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    selected = data[data.regime.eq("controlled_request") & data.method.isin(METHODS)].copy()
    # For conditioned methods retain only the certified highest-guidance row.
    selected = selected[~selected.method.str.contains("lookahead") | selected.guidance.eq(2.0)]
    if selected.groupby(["sampling_seed", "policy", "method"]).size().max() != 1:
        raise RuntimeError("duplicate method rows in aggregation")
    selected.to_csv(args.output / "selected_metrics.csv", index=False)
    summary = selected.groupby("method")[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["method" if col[0] == "method" else f"{col[0]}_{col[1]}" for col in summary.columns]
    summary.to_csv(args.output / "summary.csv", index=False)

    rng = np.random.default_rng(args.seed)
    comparisons = []
    index = selected[["sampling_seed", "policy"]].drop_duplicates().reset_index(drop=True)
    for candidate_method in [
        "lookahead_anchored_violated_t100_plus_certifier",
        "lookahead_anchored_violated_numgraph_t100_plus_certifier",
        "lookahead_anchored_violated_topk3_t100_plus_certifier",
        "lookahead_anchored_violated_topk5_t100_plus_certifier",
        "lookahead_anchored_violated_masked_t100_plus_certifier",
    ]:
        for baseline in ["empirical_marginal_bad_postrepair", "diffusion_impute_then_certify",
                         "diffusion_impute_numgraph_then_certify",
                         "diffusion_bestof8_target_then_certify",
                         "diffusion_bestof8_numgraph_then_certify", "subgroup_final_repair"]:
            if baseline not in set(selected.method):
                continue
            left = selected[selected.method.eq(candidate_method)]
            right = selected[selected.method.eq(baseline)]
            paired = left.merge(right, on=["sampling_seed", "policy"], suffixes=("_method", "_baseline"), validate="one_to_one")
            if len(paired) != len(index): raise RuntimeError("incomplete paired comparison")
            for metric in METRICS:
                delta = paired[f"{metric}_method"].to_numpy() - paired[f"{metric}_baseline"].to_numpy()
                draws = np.empty(args.bootstrap)
                for draw in range(args.bootstrap):
                    ids = rng.integers(0, len(delta), len(delta))
                    draws[draw] = delta[ids].mean()
                comparisons.append({
                    "method": candidate_method, "baseline": baseline, "metric": metric,
                    "delta_method_minus_baseline": float(delta.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
                    "pairs": len(delta),
                })
    comparison = pd.DataFrame(comparisons); comparison.to_csv(args.output / "paired_bootstrap.csv", index=False)
    lines = ["# Three-seed subgroup lookahead confirmation", "",
             "Controlled policies only. Fidelity uses heldout real rows; empirical repair draws values from train rows only.", "",
             "| Method | Validity | Active drift (SD) | Target KS | Wasserstein (SD) | Cohort numeric KS | Categorical TV | Correlation error |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.method} | {row.active_satisfaction_mean:.4f} | {row.active_drift_sd_mean:.4f} | {row.active_target_ks_reference_mean:.4f} | {row.active_target_wasserstein_sd_mean:.4f} | {row.active_cohort_numeric_ks_mean:.4f} | {row.active_cohort_categorical_tv_mean:.4f} | {row.target_context_spearman_mae_mean:.4f} |")
    lines += ["", "Bootstrap resamples nine policy-seed pairs; it is exploratory, not a substitute for more datasets."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "status": "executed", "runs": [str(run) for run in args.runs], "seeds": sorted(selected.sampling_seed.unique().tolist()),
        "regime": "controlled_request", "bootstrap_pairs": int(len(index)),
    }, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__": main()
