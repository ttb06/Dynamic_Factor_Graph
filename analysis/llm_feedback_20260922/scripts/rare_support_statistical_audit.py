#!/usr/bin/env python3
"""Paired seed-block uncertainty audit for rare-contract experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset", "catalog", "split", "seed", "contract_id"]


def block_bootstrap(frame: pd.DataFrame, rng: np.random.Generator, repeats: int) -> dict[str, tuple[float, float]]:
    seeds = frame["seed"].unique()
    values = {name: [] for name in ["speedup", "success_delta", "ks_delta", "tv_delta"]}
    for _ in range(repeats):
        selected = rng.choice(seeds, size=len(seeds), replace=True)
        sampled = pd.concat([frame[frame["seed"].eq(seed)] for seed in selected], ignore_index=True)
        values["speedup"].append(sampled["rejection_tpv"].mean() / sampled["condition_tpv"].mean())
        values["success_delta"].append(sampled["condition_success"].mean() - sampled["rejection_success"].mean())
        values["ks_delta"].append(sampled["condition_ks"].mean() - sampled["rejection_ks"].mean())
        values["tv_delta"].append(sampled["condition_tv"].mean() - sampled["rejection_tv"].mean())
    return {name: tuple(np.quantile(data, [0.025, 0.975])) for name, data in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    budgets = pd.read_csv(args.aggregate / "all_budget_curves.csv")
    metrics = pd.read_csv(args.aggregate / "all_metrics.csv")
    budgets = budgets[budgets["catalog"].str.startswith("support") & budgets["budget"].eq(1)]
    if "split" not in budgets:
        mapping = metrics[KEYS].drop_duplicates()
        budgets = budgets.merge(mapping, on=[key for key in KEYS if key != "split"], how="left", validate="many_to_one")

    rejection_budget = budgets[budgets["method"].eq("rejection")][KEYS + ["success_with_budget", "expected_trajectories_per_valid"]]
    rejection_budget = rejection_budget.rename(columns={
        "success_with_budget": "rejection_success",
        "expected_trajectories_per_valid": "rejection_tpv",
    })
    condition_budget = budgets[budgets["method"].eq("condition_g1")][KEYS + ["success_with_budget", "expected_trajectories_per_valid"]]
    condition_budget = condition_budget.rename(columns={
        "success_with_budget": "condition_success",
        "expected_trajectories_per_valid": "condition_tpv",
    })
    paired = rejection_budget.merge(condition_budget, on=KEYS, validate="one_to_one")

    rejection_metrics = metrics[metrics["method"].eq("rejection")][KEYS + ["numeric_ks", "categorical_tv"]]
    rejection_metrics = rejection_metrics.rename(columns={"numeric_ks": "rejection_ks", "categorical_tv": "rejection_tv"})
    condition_metrics = metrics[metrics["method"].eq("condition_g1_accepted")][KEYS + ["numeric_ks", "categorical_tv"]]
    condition_metrics = condition_metrics.rename(columns={"numeric_ks": "condition_ks", "categorical_tv": "condition_tv"})
    paired = paired.merge(rejection_metrics, on=KEYS, validate="one_to_one").merge(condition_metrics, on=KEYS, validate="one_to_one")
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=[
        "rejection_tpv", "condition_tpv", "rejection_ks", "condition_ks", "rejection_tv", "condition_tv"
    ])
    paired["speedup"] = paired["rejection_tpv"] / paired["condition_tpv"]
    paired["success_delta"] = paired["condition_success"] - paired["rejection_success"]
    paired["ks_delta"] = paired["condition_ks"] - paired["rejection_ks"]
    paired["tv_delta"] = paired["condition_tv"] - paired["rejection_tv"]
    paired.to_csv(args.output / "paired_seed_contract_results.csv", index=False)

    rng = np.random.default_rng(args.seed)
    rows = []
    for group_key, frame in paired.groupby(["dataset", "catalog", "split"]):
        intervals = block_bootstrap(frame, rng, args.bootstrap_repeats)
        rows.append({
            "dataset": group_key[0], "catalog": group_key[1], "split": group_key[2],
            "seeds": frame["seed"].nunique(), "contracts": frame["contract_id"].nunique(), "pairs": len(frame),
            "sampling_speedup": frame["rejection_tpv"].mean() / frame["condition_tpv"].mean(),
            "sampling_speedup_ci_low": intervals["speedup"][0], "sampling_speedup_ci_high": intervals["speedup"][1],
            "success_delta": frame["success_delta"].mean(),
            "success_delta_ci_low": intervals["success_delta"][0], "success_delta_ci_high": intervals["success_delta"][1],
            "ks_delta": frame["ks_delta"].mean(), "ks_delta_ci_low": intervals["ks_delta"][0], "ks_delta_ci_high": intervals["ks_delta"][1],
            "tv_delta": frame["tv_delta"].mean(), "tv_delta_ci_low": intervals["tv_delta"][0], "tv_delta_ci_high": intervals["tv_delta"][1],
            "sampling_win_fraction": float(frame["speedup"].gt(1).mean()),
            "ks_win_fraction": float(frame["ks_delta"].lt(0).mean()),
            "tv_win_fraction": float(frame["tv_delta"].lt(0).mean()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)

    lines = [
        "# Kiểm toán độ chắc chắn cho yêu cầu hiếm",
        "",
        "Khoảng dưới đây là bootstrap theo khối lần lặp mô hình. Số lần lặp thực tế được ghi trong từng hàng; khoảng này định lượng độ ổn định qua seed, không thay cho tái lập trên nhiều tập dữ liệu.",
        "",
        "Tỷ lệ tăng tốc lớn hơn 1 có lợi cho sinh có điều kiện. Chênh lệch KS/TV nhỏ hơn 0 có lợi cho sinh có điều kiện.",
        "",
        "| Bảng | Loại | Mức | Lần lặp | Tăng tốc lấy mẫu [95%] | Chênh lệch xác suất thành công [95%] | Chênh lệch KS [95%] | Chênh lệch TV [95%] |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.catalog} | {row.split} | {row.seeds} | "
            f"{row.sampling_speedup:.2f} [{row.sampling_speedup_ci_low:.2f}, {row.sampling_speedup_ci_high:.2f}] | "
            f"{row.success_delta:.3f} [{row.success_delta_ci_low:.3f}, {row.success_delta_ci_high:.3f}] | "
            f"{row.ks_delta:.3f} [{row.ks_delta_ci_low:.3f}, {row.ks_delta_ci_high:.3f}] | "
            f"{row.tv_delta:.3f} [{row.tv_delta_ci_low:.3f}, {row.tv_delta_ci_high:.3f}] |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
