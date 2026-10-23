#!/usr/bin/env python3
"""Convert official CuTS result arrays for any supported dataset to flat tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_METRICS = ["marginal_tv", "marginal_l2", "marginal_js", "xgb_accuracy", "xgb_balanced_accuracy", "xgb_f1"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cuts-root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--workload", default="all_three_with_labels")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    root = a.cuts_root / "experiment_data" / "constraint_program_experiments" / a.dataset / "non_dp_constraints" / "testing_results"
    suffix = f"_{a.workload}_1_1_{a.seed}_{a.epsilon}"
    rows, missing = [], []
    for base_path in sorted(root.glob(f"*{suffix}_baselines.npy")):
        program = base_path.name.removesuffix(f"{suffix}_baselines.npy")
        constrained_path = root / f"{program}{suffix}.npy"
        if not constrained_path.exists():
            missing.append(program)
            continue
        base_array = np.load(base_path)
        constrained_array = np.load(constrained_path)
        real = base_array[0, 0, 0, :, 0]
        base = base_array[1, 0, 0, :, 0]
        for param_index in range(constrained_array.shape[0]):
            value = constrained_array[param_index, 0, 0, :, 0]
            row = {"program": program, "parameter_index": param_index, "constraint_count": len(value) - len(BASE_METRICS)}
            for i, name in enumerate(BASE_METRICS):
                row[f"real_{name}"] = float(real[i])
                row[f"base_{name}"] = float(base[i])
                row[f"cuts_{name}"] = float(value[i])
            for i in range(len(value) - len(BASE_METRICS)):
                row[f"real_constraint_{i + 1}"] = float(real[len(BASE_METRICS) + i])
                row[f"base_constraint_{i + 1}"] = float(base[len(BASE_METRICS) + i])
                row[f"cuts_constraint_{i + 1}"] = float(value[len(BASE_METRICS) + i])
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(a.output / "cuts_official_long.csv", index=False)
    metadata = {"dataset": a.dataset, "programs": int(frame.program.nunique()) if len(frame) else 0, "parameter_settings": len(frame), "missing_pairs": missing}
    (a.output / "run_summary.json").write_text(json.dumps(metadata, indent=2))
    lines = [
        f"# CuTS official results on {a.dataset}", "",
        "Each row is one official hyperparameter setting. No setting is selected post hoc in this table.", "",
        "| Program | Param index | Base TV | CuTS TV | Base BAC | CuTS BAC | Base constraint 1 | CuTS constraint 1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in frame.itertuples(index=False):
        bcon = getattr(r, "base_constraint_1", float("nan"))
        ccon = getattr(r, "cuts_constraint_1", float("nan"))
        lines.append(
            f"| {r.program} | {r.parameter_index} | {r.base_marginal_tv:.4f} | {r.cuts_marginal_tv:.4f} | "
            f"{r.base_xgb_balanced_accuracy:.4f} | {r.cuts_xgb_balanced_accuracy:.4f} | {bcon:.4f} | {ccon:.4f} |"
        )
    if missing:
        lines.extend(["", "Missing constrained pairs: " + ", ".join(missing)])
    (a.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(frame.to_string(index=False))
    print(metadata)


if __name__ == "__main__":
    main()
