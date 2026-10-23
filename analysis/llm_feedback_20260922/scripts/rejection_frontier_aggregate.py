#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metric_paths = sorted(args.input_root.glob("rejection_frontier_*_seed*/frontier_metrics.csv"))
    budget_paths = sorted(args.input_root.glob("rejection_frontier_*_seed*/budget_curve.csv"))
    if not metric_paths or not budget_paths:
        raise FileNotFoundError("rejection frontier outputs not found")
    metrics = pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in metric_paths], ignore_index=True)
    budgets = pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in budget_paths], ignore_index=True)
    if "split" not in budgets.columns or budgets["split"].isna().any():
        contract_splits = metrics[["dataset", "catalog", "seed", "contract_id", "split"]].drop_duplicates()
        budgets = budgets.merge(
            contract_splits.rename(columns={"split": "split_from_metrics"}),
            on=["dataset", "catalog", "seed", "contract_id"], how="left", validate="many_to_one",
        )
        if "split" not in budgets.columns:
            budgets["split"] = budgets["split_from_metrics"]
        else:
            budgets["split"] = budgets["split"].fillna(budgets["split_from_metrics"])
        budgets = budgets.drop(columns="split_from_metrics")
    metrics.to_csv(args.output / "all_metrics.csv", index=False)
    budgets.to_csv(args.output / "all_budget_curves.csv", index=False)
    summary = metrics.groupby(["dataset", "catalog", "method"]).agg(
        seeds=("seed", "nunique"), contracts=("contract_id", "nunique"), satisfaction=("satisfaction", "mean"),
        numeric_ks=("numeric_ks", "mean"), categorical_tv=("categorical_tv", "mean"), edit_rate=("edit_rate", "mean"),
        numeric_distortion=("numeric_distortion", "mean")
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    budget_summary = budgets.groupby(["dataset", "catalog", "method", "budget"]).agg(
        seeds=("seed", "nunique"), contracts=("contract_id", "nunique"), heldout_support=("heldout_support", "mean"),
        success=("success_with_budget", "mean"), trajectories_per_valid=("expected_trajectories_per_valid", "mean")
    ).reset_index()
    budget_summary.to_csv(args.output / "budget_summary.csv", index=False)
    stratified_summary = metrics.groupby(["dataset", "catalog", "split", "method"]).agg(
        seeds=("seed", "nunique"), contracts=("contract_id", "nunique"), satisfaction=("satisfaction", "mean"),
        numeric_ks=("numeric_ks", "mean"), categorical_tv=("categorical_tv", "mean"), edit_rate=("edit_rate", "mean"),
        numeric_distortion=("numeric_distortion", "mean")
    ).reset_index()
    stratified_summary.to_csv(args.output / "stratified_summary.csv", index=False)
    stratified_budget_summary = budgets.groupby(["dataset", "catalog", "split", "method", "budget"]).agg(
        seeds=("seed", "nunique"), contracts=("contract_id", "nunique"), heldout_support=("heldout_support", "mean"),
        success=("success_with_budget", "mean"), trajectories_per_valid=("expected_trajectories_per_valid", "mean")
    ).reset_index()
    stratified_budget_summary.to_csv(args.output / "stratified_budget_summary.csv", index=False)
    lines = [
        "# Đường biên giữa lọc mẫu, sinh có điều kiện và chứng nhận",
        "",
        "Phân phối tham chiếu lấy từ tập test. Các hợp đồng là yêu cầu cohort được kiểm soát, không phải quy luật miền.",
        "",
        "| Bảng | Danh mục | Phương pháp | Thỏa yêu cầu | KS số | TV phân loại | Tỷ lệ sửa | Biến dạng số |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.catalog} | {row.method} | {row.satisfaction:.4f} | {row.numeric_ks:.4f} | {row.categorical_tv:.4f} | {row.edit_rate:.4f} | {row.numeric_distortion:.4f} |")
    lines.extend([
        "",
        "## Tách riêng theo mức độ hiếm",
        "",
        "Bảng dưới đây chỉ hiện hai đối chứng chính ở ngân sách một lượt sinh. `split` ghi phân vị dùng để tạo yêu cầu từ tập train.",
        "",
        "| Bảng | Danh mục | Mức | Phương pháp | Hỗ trợ test | Xác suất thành công | Lượt sinh / mẫu hợp lệ |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ])
    focus = stratified_budget_summary[
        stratified_budget_summary["catalog"].str.startswith("support")
        & stratified_budget_summary["method"].isin(["rejection", "condition_g1"])
        & stratified_budget_summary["budget"].eq(1)
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.catalog} | {row.split} | {row.method} | "
            f"{row.heldout_support:.4f} | {row.success:.4f} | {row.trajectories_per_valid:.2f} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"loaded {len(metric_paths)} runs")


if __name__ == "__main__":
    main()
