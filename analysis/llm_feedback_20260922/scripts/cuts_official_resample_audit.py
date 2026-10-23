#!/usr/bin/env python3
"""Audit CuTS sampling variability for the official Adult programs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["marginal_tv", "marginal_l2", "marginal_js", "xgb_accuracy", "xgb_balanced_accuracy", "xgb_f1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.cuts_root / "experiment_data/constraint_program_experiments/ADULT/non_dp_constraints/testing_results"
    rows: list[dict[str, float | int | str]] = []
    suffix = "_all_three_with_labels_1_5_42_0.0.npy"
    for path in sorted(root.glob(f"*{suffix}")):
        if path.name.endswith("_baselines.npy"):
            continue
        program = path.name.removesuffix(suffix)
        base_path = root / f"{program}_all_three_with_labels_1_5_42_0.0_baselines.npy"
        if not base_path.exists():
            continue
        cuts = np.load(path)[0, 0, :, :, 0]
        bases = np.load(base_path)[1, 0, :, :, 0]
        for method, array in (("base", bases), ("CuTS", cuts)):
            for resample, values in enumerate(array):
                row: dict[str, float | int | str] = {
                    "program": program,
                    "method": method,
                    "resample": resample,
                }
                for index, name in enumerate(METRICS):
                    row[name] = float(values[index])
                for index in range(max(0, len(values) - len(METRICS))):
                    row[f"constraint_{index + 1}"] = float(values[len(METRICS) + index])
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError("No matched CuTS 1x5 result pairs")
    frame.to_csv(args.output / "per_resample.csv", index=False)
    focus_names = ["implication2", "implication4", "implication5", "line_constraint1", "line_constraint2"]
    focus = frame[frame["program"].isin(focus_names)]
    summary = focus.groupby(["program", "method"]).agg(
        resamples=("resample", "nunique"), tv_mean=("marginal_tv", "mean"), tv_sd=("marginal_tv", "std"),
        bac_mean=("xgb_balanced_accuracy", "mean"), bac_sd=("xgb_balanced_accuracy", "std"),
        validity_mean=("constraint_1", "mean"), validity_sd=("constraint_1", "std"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    lines = [
        "# Độ biến thiên khi lấy mẫu của CuTS trên Adult",
        "",
        "Các lần lặp ở đây dùng cùng mô hình và cùng lần tinh chỉnh; chúng chỉ đo độ biến thiên khi lấy mẫu, không thay thế các seed huấn luyện độc lập.",
        "",
        "| Chương trình | Phương pháp | Lần lấy mẫu | TV | BAC | Điểm ràng buộc |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.program} | {row.method} | {row.resamples} | {row.tv_mean:.4f} ± {row.tv_sd:.4f} | "
            f"{row.bac_mean:.4f} ± {row.bac_sd:.4f} | {row.validity_mean:.4f} ± {row.validity_sd:.4f} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
