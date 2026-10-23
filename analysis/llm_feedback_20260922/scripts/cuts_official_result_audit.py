#!/usr/bin/env python3
"""Summarize official CuTS arrays into a readable matched baseline table."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_METRICS = ["marginal_tv", "marginal_l2", "marginal_js", "xgb_accuracy", "xgb_balanced_accuracy", "xgb_f1"]


def vector(path: Path, baseline_index: int | None) -> np.ndarray:
    data = np.load(path)
    if baseline_index is None:
        return data[0, 0, 0, :, 0]
    return data[baseline_index, 0, 0, :, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result_root = args.cuts_root / "experiment_data/constraint_program_experiments/ADULT/non_dp_constraints/testing_results"
    rows = []
    for baseline_path in sorted(result_root.glob("*_baselines.npy")):
        stem = baseline_path.name.removesuffix("_all_three_with_labels_1_1_42_0.0_baselines.npy")
        constrained_path = result_root / f"{stem}_all_three_with_labels_1_1_42_0.0.npy"
        if not constrained_path.exists():
            continue
        real = vector(baseline_path, 0)
        base = vector(baseline_path, 1)
        constrained = vector(constrained_path, None)
        constraint_count = max(0, len(constrained) - len(BASE_METRICS))
        row = {"program": stem, "constraint_count": constraint_count}
        for index, name in enumerate(BASE_METRICS):
            row[f"real_{name}"] = float(real[index])
            row[f"base_{name}"] = float(base[index])
            row[f"cuts_{name}"] = float(constrained[index])
            row[f"cuts_minus_base_{name}"] = float(constrained[index] - base[index])
        for index in range(constraint_count):
            row[f"real_constraint_{index + 1}"] = float(real[len(BASE_METRICS) + index])
            row[f"base_constraint_{index + 1}"] = float(base[len(BASE_METRICS) + index])
            row[f"cuts_constraint_{index + 1}"] = float(constrained[len(BASE_METRICS) + index])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "cuts_official_results.csv", index=False)
    focus_names = ["implication2", "implication4", "implication5", "line_constraint1", "line_constraint2"]
    focus = frame[frame["program"].isin(focus_names)].copy()
    lines = [
        "# Đối chứng CuTS chính thức trên Adult",
        "",
        "Mỗi tệp hiện có một lần huấn luyện/lấy mẫu. `base` là checkpoint CuTS chưa tinh chỉnh; `CuTS` là kết quả sau tinh chỉnh theo chương trình. Với implication và row constraint, mã chính thức còn dùng lọc mẫu.",
        "",
        "| Chương trình | TV nền | TV CuTS | ΔTV | BAC nền | BAC CuTS | Điểm ràng buộc nền | Điểm ràng buộc CuTS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.program} | {row.base_marginal_tv:.4f} | {row.cuts_marginal_tv:.4f} | "
            f"{row.cuts_minus_base_marginal_tv:+.4f} | {row.base_xgb_balanced_accuracy:.4f} | "
            f"{row.cuts_xgb_balanced_accuracy:.4f} | {getattr(row, 'base_constraint_1', float('nan')):.4f} | "
            f"{getattr(row, 'cuts_constraint_1', float('nan')):.4f} |"
        )
    lines.extend([
        "",
        "Thời gian quan sát từ Slurm: baseline toàn bộ chương trình 4 phút 30 giây; CuTS có tinh chỉnh 1 giờ 57 phút 22 giây.",
        "Các con số này là kết quả đã chạy nhưng mới có một seed; chưa dùng để tuyên bố hơn/kém CuTS.",
    ])
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
