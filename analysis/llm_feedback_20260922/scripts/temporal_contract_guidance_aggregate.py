#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metric_paths = sorted(args.input_root.glob("temporal_contract_guidance_*_seed*/temporal_metrics.csv"))
    budget_paths = sorted(args.input_root.glob("temporal_contract_guidance_*_seed*/temporal_budget.csv"))
    if not metric_paths or not budget_paths:
        raise FileNotFoundError("temporal guidance outputs not found")
    metrics = pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in metric_paths], ignore_index=True)
    budgets = pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in budget_paths], ignore_index=True)
    # A few queued jobs may have loaded an intermediate method label while the
    # target-scoped extension was being installed.  ``all`` is the historical
    # default, so normalize that transient suffix before grouping.
    metrics["method"] = metrics["method"].str.replace(r"_all(?=(_accepted|_certified)?$)", "", regex=True)
    budgets["method"] = budgets["method"].str.replace(r"_all$", "", regex=True)
    metrics.to_csv(args.output / "all_metrics.csv", index=False)
    budgets.to_csv(args.output / "all_budgets.csv", index=False)
    summary = metrics.groupby([
        "dataset", "catalog", "split", "profile", "peak", "floor", "cutoff", "method"
    ]).agg(
        seeds=("seed", "nunique"), contracts=("contract_id", "nunique"), satisfaction=("satisfaction", "mean"),
        numeric_ks=("numeric_ks", "mean"), categorical_tv=("categorical_tv", "mean"),
        edit_rate=("edit_rate", "mean"), numeric_distortion=("numeric_distortion", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    budget_summary = budgets.groupby([
        "dataset", "catalog", "split", "profile", "peak", "floor", "cutoff", "method"
    ]).agg(
        seeds=("seed", "nunique"), contracts=("contract_id", "nunique"), heldout_support=("heldout_support", "mean"),
        success=("single_trajectory_success", "mean"),
        trajectories_per_valid=("expected_trajectories_per_valid", "mean"),
        denoiser_calls_per_valid=("expected_denoiser_calls_per_valid", "mean"),
    ).reset_index()
    budget_summary.to_csv(args.output / "budget_summary.csv", index=False)

    accepted = summary[summary["method"].str.endswith("_accepted")].copy()
    raw = budget_summary.copy()
    accepted["raw_method"] = accepted["method"].str.removesuffix("_accepted")
    paired = accepted.merge(
        raw,
        left_on=["dataset", "catalog", "split", "profile", "peak", "floor", "cutoff", "raw_method"],
        right_on=["dataset", "catalog", "split", "profile", "peak", "floor", "cutoff", "method"],
        suffixes=("_fidelity", "_budget"), validate="one_to_one",
    )
    paired["combined_error"] = paired["numeric_ks"] + paired["categorical_tv"]
    paired["pareto"] = False
    for _, index in paired.groupby(["dataset", "catalog", "split"]).groups.items():
        block = paired.loc[index]
        for row_index, row in block.iterrows():
            dominated = (
                (block["denoiser_calls_per_valid"] <= row["denoiser_calls_per_valid"])
                & (block["combined_error"] <= row["combined_error"])
                & ((block["denoiser_calls_per_valid"] < row["denoiser_calls_per_valid"])
                   | (block["combined_error"] < row["combined_error"]))
            ).any()
            paired.loc[row_index, "pareto"] = not dominated
    paired.to_csv(args.output / "pareto.csv", index=False)
    best = paired[paired["pareto"]].sort_values(["dataset", "catalog", "split", "denoiser_calls_per_valid", "combined_error"])
    lines = [
        "# Dẫn hướng hợp đồng thay đổi theo thời gian",
        "",
        f"Đã đọc {len(metric_paths)} lần chạy. Bảng chỉ hiện các cấu hình không bị cấu hình khác trội đồng thời về chi phí và sai khác phân phối.",
        "",
        "| Bảng | Loại | Mức | Lịch | Đỉnh | Sàn | Mốc | Thành công | Lần gọi bộ khử nhiễu/mẫu hợp lệ | KS | TV |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.catalog} | {row.split} | {row.profile} | {row.peak:g} | {row.floor:g} | "
            f"{row.cutoff:g} | {row.success:.4f} | {row.denoiser_calls_per_valid:.2f} | {row.numeric_ks:.4f} | {row.categorical_tv:.4f} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
