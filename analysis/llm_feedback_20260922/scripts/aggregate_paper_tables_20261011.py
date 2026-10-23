#!/usr/bin/env python3
"""Aggregate the frozen three-dataset confirmation runs used in the paper report.

The script intentionally names every selected run.  Development sweeps are not
searched or ranked, so rerunning it cannot silently replace a confirmation
result with a better tuning run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DYNAMIC = [
    "graph_rate_expanded_multi_entry_v3_graph_rate_1067515_1",
    "graph_rate_8edge_s20261377_graph_rate_1067520_1",
    "graph_rate_8edge_s20261378_graph_rate_1067520_3",
]
DEFAULT_FROZEN = [
    "default_8edge_frozen_s20261376_1067698_0",
    "default_8edge_frozen_s20261377_1067698_1",
    "default_8edge_frozen_s20261378_1067698_2",
]
DEFAULT_HARPOON = [
    "harpoon_default_8edge_confirm_s20261376_1067613_0",
    "harpoon_default_8edge_confirm_s20261377_1067613_1",
    "harpoon_default_8edge_confirm_s20261378_1067613_2",
]
SHOPPERS_DYNAMIC = [
    "shoppers_promote_confirm_s20261383_dynamic_1067743_0",
    "shoppers_promote_confirm_s20261384_dynamic_1067743_2",
    "shoppers_promote_confirm_s20261385_dynamic_1067743_4",
]
SHOPPERS_FROZEN = [
    "shoppers_promote_confirm_s20261383_frozen_state_1067743_1",
    "shoppers_promote_confirm_s20261384_frozen_state_1067743_3",
    "shoppers_promote_confirm_s20261385_frozen_state_1067743_5",
]
DIABETES_EXPANDED = [
    "diabetes_expanded_confirm_s20261383_dynamic_1068457_0",
    "diabetes_expanded_confirm_s20261384_dynamic_1068457_1",
    "diabetes_expanded_confirm_s20261385_dynamic_1068457_2",
]
DEFAULT_COMMON = [
    "harpoon_dynamic_common_s20261376_1067619_0",
    "harpoon_dynamic_common_s20261377_1067619_1",
    "harpoon_dynamic_common_s20261378_1067619_2",
]
SHOPPERS_COMMON = [
    "shoppers_promoted_common_s20261383_1068460_0",
    "shoppers_promoted_common_s20261384_1068460_1",
    "shoppers_promoted_common_s20261385_1068460_2",
]
DIABETES_COMMON = [
    "diabetes_expanded_common_s20261383_1068463_0",
    "diabetes_expanded_common_s20261384_1068463_1",
    "diabetes_expanded_common_s20261385_1068463_2",
]


def load_method(outputs: Path, run: str, method: str, dataset: str, label: str, seed: int) -> dict:
    path = outputs / run / "method_metrics.csv"
    frame = pd.read_csv(path).set_index("method")
    row = frame.loc[method].to_dict()
    return {"dataset": dataset, "seed": seed, "method": label, "run": run, **row}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = args.analysis / "outputs"
    rows: list[dict] = []

    for offset, (dynamic, frozen, harpoon) in enumerate(zip(DEFAULT_DYNAMIC, DEFAULT_FROZEN, DEFAULT_HARPOON)):
        seed = 20261376 + offset
        rows += [
            load_method(outputs, dynamic, "unconditional", "Default", "TabDiff", seed),
            load_method(outputs, dynamic, "final_reward_calibrated", "Default", "Final repair", seed),
            load_method(outputs, dynamic, "xstate_reward_calibrated", "Default", "Dynamic trajectory", seed),
            load_method(outputs, frozen, "xstate_reward_calibrated", "Default", "Frozen trajectory", seed),
        ]
        harpoon_row = pd.read_csv(outputs / harpoon / "method_metrics.csv").set_index("method").loc[
            "harpoon_hard_w8"
        ].to_dict()
        rows.append({"dataset": "Default", "seed": seed, "method": "Harpoon-style proxy", "run": harpoon,
                     **harpoon_row})

    for offset, (dynamic, frozen) in enumerate(zip(SHOPPERS_DYNAMIC, SHOPPERS_FROZEN)):
        seed = 20261383 + offset
        rows += [
            load_method(outputs, dynamic, "unconditional", "Shoppers", "TabDiff", seed),
            load_method(outputs, dynamic, "final_promoted_reward_calibrated", "Shoppers", "Final repair", seed),
            load_method(outputs, dynamic, "x0_reward_calibrated", "Shoppers", "Dynamic trajectory", seed),
            load_method(outputs, dynamic, "x0_reward_calibrated_then_final_promoted_reward",
                        "Shoppers", "Dynamic + residual repair", seed),
            load_method(outputs, frozen, "x0_reward_calibrated", "Shoppers", "Frozen trajectory", seed),
            load_method(outputs, frozen, "x0_reward_calibrated_then_final_promoted_reward",
                        "Shoppers", "Frozen + residual repair", seed),
        ]

    for offset, run in enumerate(DIABETES_EXPANDED):
        seed = 20261383 + offset
        rows += [
            load_method(outputs, run, "unconditional", "Diabetes", "TabDiff", seed),
            load_method(outputs, run, "final_promoted_reward_calibrated", "Diabetes", "Final repair", seed),
            load_method(outputs, run, "x0_reward_calibrated", "Diabetes", "Dynamic trajectory", seed),
            load_method(outputs, run, "x0_reward_calibrated_then_final_promoted_reward",
                        "Diabetes", "Dynamic + residual repair", seed),
        ]

    raw = pd.DataFrame(rows)
    raw.to_csv(args.output / "main_results_by_seed.csv", index=False)
    metrics = ["heldout_violation_error", "calibration_violation_error", "macro_violation",
               "numeric_ks", "categorical_tv", "affected_numeric_drift_sd", "action_events",
               "runtime_seconds"]
    available = [column for column in metrics if column in raw.columns]
    summary = raw.groupby(["dataset", "method"])[available].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(args.output / "main_results_summary.csv", index=False)
    print(summary.to_string())

    common_rows: list[pd.DataFrame] = []
    common_specs = [
        ("Default", 20261376, DEFAULT_COMMON,
         {"continued_unguided": "TabDiff", "final_repair": "Final repair",
          "dynamic_graph_rate": "Dynamic trajectory", "harpoon_hard_w8": "Harpoon-style proxy"}),
        ("Shoppers", 20261383, SHOPPERS_COMMON,
         {"unconditional": "TabDiff", "final_promoted": "Final repair",
          "frozen_promoted": "Frozen + residual repair",
          "dynamic_promoted": "Dynamic + residual repair"}),
        ("Diabetes", 20261383, DIABETES_COMMON,
         {"unconditional": "TabDiff", "final_promoted": "Final repair",
          "trajectory_only": "Dynamic trajectory", "dynamic_full": "Dynamic + residual repair"}),
    ]
    for dataset, first_seed, runs, labels in common_specs:
        for offset, run in enumerate(runs):
            frame = pd.read_csv(outputs / run / "baseline_metrics.csv")
            frame = frame[frame["method"].isin(labels)].copy()
            frame["method"] = frame["method"].map(labels)
            frame.insert(0, "dataset", dataset)
            frame.insert(1, "seed", first_seed + offset)
            frame.insert(3, "run", run)
            common_rows.append(frame)
    common = pd.concat(common_rows, ignore_index=True)
    common.to_csv(args.output / "common_quality_by_seed.csv", index=False)
    common_metrics = ["shape_error_percent", "trend_error_percent", "alpha_precision_percent",
                      "beta_recall_percent", "detection_score", "mle_xgboost_auc",
                      "dcr_percent_50_ideal"]
    common_summary = common.groupby(["dataset", "method"])[common_metrics].agg(["mean", "std"])
    common_summary.columns = ["_".join(column) for column in common_summary.columns]
    common_summary.reset_index().to_csv(args.output / "common_quality_summary.csv", index=False)


if __name__ == "__main__":
    main()
