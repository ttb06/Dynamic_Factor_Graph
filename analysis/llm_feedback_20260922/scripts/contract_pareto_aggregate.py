#!/usr/bin/env python3
"""Combine three-seed adapter and actuation baselines into budgeted Pareto tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value): values.append("")
            elif isinstance(value, float): values.append(f"{value:.4f}")
            else: values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-by-seed", type=Path, required=True)
    parser.add_argument("--actuation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    adapter = pd.read_csv(args.adapter_by_seed)
    adapter = adapter.rename(columns={"cohort_ks": "numeric_ks", "categorical_tv": "categorical_tv"})
    adapter["method"] = adapter.apply(lambda row: f"adapter_{row.representation}_cfg{int(row.guidance_scale)}", axis=1)
    adapter["passes"] = adapter.guidance_scale.where(adapter.guidance_scale > 1, 1)
    adapter_rows = adapter[["method", "seed", "split", "satisfaction", "numeric_ks", "categorical_tv", "coverage", "passes"]]
    actuation_frames = []
    for path in sorted(args.actuation_root.glob("contract_actuation_*/contract_metrics.csv")):
        frame = pd.read_csv(path)
        grouped = frame.groupby(["method", "seed", "split"]).agg(
            satisfaction=("satisfaction_rate", "mean"), numeric_ks=("cohort_numeric_ks_mean", "mean"),
            categorical_tv=("cohort_categorical_tv_mean", "mean"), coverage=("cohort_category_coverage_mean", "mean"),
            passes=("denoiser_passes_per_output", "mean"),
        ).reset_index()
        actuation_frames.append(grouped)
    actuation = pd.concat(actuation_frames, ignore_index=True)
    all_rows = pd.concat([adapter_rows, actuation], ignore_index=True)
    all_rows.to_csv(args.output / "comparison_by_seed.csv", index=False)
    summary = all_rows.groupby(["method", "split"]).agg(
        seeds=("seed", "nunique"), satisfaction_mean=("satisfaction", "mean"), satisfaction_std=("satisfaction", "std"),
        numeric_ks_mean=("numeric_ks", "mean"), numeric_ks_std=("numeric_ks", "std"),
        categorical_tv_mean=("categorical_tv", "mean"), coverage_mean=("coverage", "mean"), passes=("passes", "mean"),
    ).reset_index()
    pareto: dict[int, bool] = {}
    for split, group in summary.groupby("split"):
        for index, row in group.iterrows():
            dominated = False
            for other_index, other in group.iterrows():
                if index == other_index: continue
                no_worse = (
                    other.satisfaction_mean >= row.satisfaction_mean
                    and other.numeric_ks_mean <= row.numeric_ks_mean
                    and other.categorical_tv_mean <= row.categorical_tv_mean
                    and other.passes <= row.passes
                )
                strict = (
                    other.satisfaction_mean > row.satisfaction_mean
                    or other.numeric_ks_mean < row.numeric_ks_mean
                    or other.categorical_tv_mean < row.categorical_tv_mean
                    or other.passes < row.passes
                )
                if no_worse and strict: dominated = True; break
            pareto[index] = not dominated
    summary["pareto_nondominated"] = summary.index.map(pareto)
    summary.to_csv(args.output / "pareto_summary.csv", index=False)
    key = summary[summary.method.isin([
        "unconditional", "final_projection", "best_of_2", "best_of_4",
        "adapter_structured_cfg1", "adapter_structured_cfg2", "adapter_text_cfg2", "adapter_hybrid_cfg2",
    ])]
    (args.output / "RESULTS.md").write_text(
        "# Budgeted actuator Pareto audit\n\nThree seeds; satisfaction is maximized while cohort numeric KS, categorical TV, and denoiser passes are minimized.\n\n"
        + markdown_table(key) + "\n"
    )
    (args.output / "run_metadata.json").write_text(json.dumps({"status": "executed", "seeds": sorted(all_rows.seed.unique().tolist()), "rows": len(all_rows)}, indent=2))
    print(key.to_string(index=False))


if __name__ == "__main__": main()
