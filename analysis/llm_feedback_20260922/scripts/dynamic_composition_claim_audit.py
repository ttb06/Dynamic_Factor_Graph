#!/usr/bin/env python3
"""Focused claim audit for non-rarity mixed/compositional dynamic graph experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"temporal_contract_guidance_*/{filename}"))
    frames = [pd.read_csv(path).assign(source=str(path.parent)) for path in paths]
    if not frames:
        raise FileNotFoundError(filename)
    return pd.concat(frames, ignore_index=True)


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metrics = read(args.input_root, "temporal_metrics.csv")
    budgets = read(args.input_root, "temporal_budget.csv")
    metrics = metrics[metrics["catalog"].eq("mixed") & metrics["seed"].eq(20260925)].copy()
    budgets = budgets[budgets["catalog"].eq("mixed") & budgets["seed"].eq(20260925)].copy()
    # Retries and partial analyses can coexist.  Keep the newest complete run.
    metric_key = ["dataset", "contract_id", "profile", "peak", "floor", "actuation_scope", "method"]
    budget_key = ["dataset", "contract_id", "profile", "peak", "floor", "actuation_scope"]
    metrics = metrics.sort_values("source").drop_duplicates(metric_key, keep="last")
    budgets = budgets.sort_values("source").drop_duplicates(budget_key, keep="last")

    accepted = metrics[metrics["method"].str.endswith("_accepted")].copy()
    accepted["variant"] = np.where(
        accepted["actuation_scope"].eq("dynamic_graph"), "dynamic",
        np.where(accepted["actuation_scope"].eq("target_neighbors"), "fixed", "other"),
    )
    accepted = accepted[accepted["variant"].isin(["dynamic", "fixed"])]
    accepted = accepted.merge(
        budgets[["dataset", "contract_id", "profile", "peak", "floor", "actuation_scope",
                 "single_trajectory_success", "expected_denoiser_calls_per_valid"]],
        on=["dataset", "contract_id", "profile", "peak", "floor", "actuation_scope"],
        validate="one_to_one",
    )
    columns = ["dataset", "split", "contract_id", "profile", "peak", "floor", "variant",
               "single_trajectory_success", "expected_denoiser_calls_per_valid", "numeric_ks", "categorical_tv"]
    wide = accepted[columns].pivot(
        index=["dataset", "split", "contract_id", "profile", "peak", "floor"],
        columns="variant",
    ).reset_index()
    wide.columns = ["_".join(str(x) for x in item if str(x)) for item in wide.columns]
    for metric in ["single_trajectory_success", "expected_denoiser_calls_per_valid", "numeric_ks", "categorical_tv"]:
        wide[f"delta_{metric}"] = wide[f"{metric}_dynamic"] - wide[f"{metric}_fixed"]
    wide.to_csv(args.output / "dynamic_vs_fixed.csv", index=False)

    # Compare dynamic guidance plus residual certification against repair-only
    # (peak=0 unconditional trajectory plus the identical certifier).
    certified = metrics[metrics["method"].str.endswith("_certified")].copy()
    dynamic_cert = certified[
        certified["actuation_scope"].eq("dynamic_graph") & certified["peak"].eq(4)
    ].copy()
    repair_only = certified[certified["peak"].eq(0)].copy()
    repair_only = repair_only.sort_values("source").drop_duplicates(["dataset", "contract_id"], keep="last")
    comparison = dynamic_cert.merge(
        repair_only[["dataset", "contract_id", "numeric_ks", "categorical_tv", "edit_rate", "numeric_distortion"]],
        on=["dataset", "contract_id"], suffixes=("_dynamic_cert", "_repair_only"), validate="many_to_one",
    )
    for metric in ["numeric_ks", "categorical_tv", "edit_rate", "numeric_distortion"]:
        comparison[f"delta_{metric}"] = comparison[f"{metric}_dynamic_cert"] - comparison[f"{metric}_repair_only"]
    comparison.to_csv(args.output / "dynamic_certifier_vs_repair_only.csv", index=False)

    rng = np.random.default_rng(20260929)
    rows = []
    for values, block in wide.groupby(["dataset", "split", "profile"]):
        row = dict(zip(["dataset", "split", "profile"], values))
        row["contracts"] = len(block)
        for metric in ["delta_single_trajectory_success", "delta_expected_denoiser_calls_per_valid",
                       "delta_numeric_ks", "delta_categorical_tv"]:
            x = block[metric].to_numpy(float)
            row[metric] = float(np.nanmean(x))
            row[metric + "_ci_low"], row[metric + "_ci_high"] = bootstrap(x, rng)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)

    lines = [
        "# Dynamic graph on non-rarity mixed and compositional conditions", "",
        "One seed is used for fast hypothesis screening. Conditions come from held-out thresholds and compositions, not the quantile-rarity sweep.", "",
        "## Dynamic versus fixed dependency graph", "",
        "| Dataset | Split | Schedule | Contracts | Δ success | Δ calls/valid | Δ KS | Δ TV |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.split} | {row.profile} | {row.contracts} | "
            f"{row.delta_single_trajectory_success:+.4f} | "
            f"{row.delta_expected_denoiser_calls_per_valid:+.2f} | "
            f"{row.delta_numeric_ks:+.4f} | {row.delta_categorical_tv:+.4f} |"
        )
    lines += ["", "## Dynamic guidance plus certifier versus repair-only", ""]
    if len(comparison):
        for metric in ["edit_rate", "numeric_distortion", "numeric_ks", "categorical_tv"]:
            value = float(comparison[f"delta_{metric}"].mean())
            lines.append(f"- Mean Δ {metric}: `{value:+.4f}` (negative favors dynamic guidance before certification).")
    else:
        lines.append("Repair-only controls were not available.")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
