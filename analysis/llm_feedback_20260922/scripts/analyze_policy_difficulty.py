#!/usr/bin/env python3
"""Summarize policy-level wins without hiding heterogeneous thresholds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METHODS = [
    "lookahead_anchored_violated_masked_t100_plus_certifier",
    "lookahead_anchored_violated_topk3_t100_plus_certifier",
    "lookahead_anchored_violated_topk5_t100_plus_certifier",
    "lookahead_anchored_violated_numgraph_t100_plus_certifier",
    "lookahead_anchored_violated_t100_plus_certifier",
]
BASELINES = [
    "diffusion_impute_then_certify",
    "diffusion_impute_numgraph_then_certify",
    "empirical_marginal_bad_postrepair",
]
METRICS = [
    "active_drift_sd",
    "active_target_ks_reference",
    "active_target_wasserstein_sd",
    "active_cohort_numeric_ks",
    "active_cohort_categorical_tv",
    "target_context_spearman_mae",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--shoppers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    comparisons = []
    for dataset, path in [("default", args.default), ("shoppers", args.shoppers)]:
        data = pd.read_csv(path)
        data = data[data.regime.eq("controlled_request")].copy()
        unconditional = data[data.method.eq("unconditional")][["policy", "active_satisfaction"]].rename(
            columns={"active_satisfaction": "unconditional_satisfaction"})
        for method in METHODS:
            candidate = data[data.method.eq(method)].merge(unconditional, on="policy", validate="one_to_one")
            if candidate.empty:
                continue
            for baseline in BASELINES:
                reference = data[data.method.eq(baseline)]
                if reference.empty:
                    continue
                paired = candidate.merge(reference, on=["policy", "regime", "threshold"], suffixes=("_method", "_baseline"), validate="one_to_one")
                for row in paired.itertuples(index=False):
                    record = {
                        "dataset": dataset,
                        "policy": row.policy,
                        "threshold": row.threshold,
                        "method": method,
                        "baseline": baseline,
                        "unconditional_violation": 1.0 - row.unconditional_satisfaction,
                    }
                    for metric in METRICS:
                        method_value = getattr(row, metric + "_method")
                        baseline_value = getattr(row, metric + "_baseline")
                        record[metric + "_method"] = method_value
                        record[metric + "_baseline"] = baseline_value
                        record[metric + "_gain"] = baseline_value - method_value
                    comparisons.append(record)
    result = pd.DataFrame(comparisons)
    result.to_csv(args.output / "policy_level_comparisons.csv", index=False)

    rows = []
    for (dataset, method, baseline), group in result.groupby(["dataset", "method", "baseline"]):
        summary = {"dataset": dataset, "method": method, "baseline": baseline, "policies": len(group)}
        for metric in METRICS:
            gain = group[metric + "_gain"]
            summary[metric + "_wins"] = int((gain > 0).sum())
            summary[metric + "_mean_gain"] = float(gain.mean())
            correlation = spearmanr(group.unconditional_violation, gain).statistic
            summary[metric + "_difficulty_rho"] = float(correlation) if np.isfinite(correlation) else np.nan
        gains = group[[metric + "_gain" for metric in METRICS]]
        summary["strictly_dominates"] = int(((gains >= 0).all(axis=1) & (gains > 0).any(axis=1)).sum())
        summary["strictly_dominated"] = int(((gains <= 0).all(axis=1) & (gains < 0).any(axis=1)).sum())
        rows.append(summary)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)

    lines = ["# Policy difficulty audit", "",
             "Positive gain means the listed method has lower error or displacement than its baseline.", "",
             "| Dataset | Method | Baseline | Policies | Drift wins | Target KS wins | Wasserstein wins | Numeric KS wins | Categorical TV wins | Correlation wins | Dominates | Dominated |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.method} | {row.baseline} | {row.policies} | {row.active_drift_sd_wins} | "
            f"{row.active_target_ks_reference_wins} | {row.active_target_wasserstein_sd_wins} | "
            f"{row.active_cohort_numeric_ks_wins} | {row.active_cohort_categorical_tv_wins} | "
            f"{row.target_context_spearman_mae_wins} | {row.strictly_dominates} | {row.strictly_dominated} |"
        )
    lines += ["", "Difficulty correlations are in `summary.csv`; this is a one-seed exploratory sweep."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
