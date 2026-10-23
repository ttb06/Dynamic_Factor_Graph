#!/usr/bin/env python3
"""Aggregate transition-certifier probes and select wait on calibration seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.groupby(["dataset", "state_kind", "wait", "direction"], sort=False).agg(
        raw_flips=("raw_flips", "sum"),
        raw_final_correct=("raw_final_correct", "sum"),
        accepted=("accepted", "sum"),
        accepted_final_correct=("accepted_final_correct", "sum"),
    ).reset_index()
    result["precision"] = result.accepted_final_correct / result.accepted.replace(0, np.nan)
    result["recall"] = result.accepted_final_correct / result.raw_final_correct.replace(0, np.nan)
    result["acceptance_rate"] = result.accepted / result.raw_flips.replace(0, np.nan)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=.95)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    paths = sorted(args.root.glob(f"transition_survival_*_{args.job_id}_*/transition_events.csv"))
    if len(paths) != 9:
        raise RuntimeError(f"expected 9 transition files, found {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame[(frame.direction == "all") & frame.source_time_bin.isin(["(0.50,0.75]", "(0.25,0.50]", "(0.10,0.25]"])].copy()
    minimum_seed = frame.groupby("dataset").seed.transform("min")
    calibration = frame[frame.seed == minimum_seed]
    evaluation = frame[frame.seed != minimum_seed]
    calibration_summary = summarize(calibration)
    evaluation_summary = summarize(evaluation)
    calibration_summary.to_csv(args.output / "calibration_summary.csv", index=False)
    evaluation_summary.to_csv(args.output / "evaluation_summary.csv", index=False)

    selections: list[dict] = []
    for (dataset, state_kind), block in calibration_summary.groupby(["dataset", "state_kind"]):
        eligible = block[(block.precision >= args.target_precision) & (block.accepted > 0)]
        if eligible.empty:
            selected = block.sort_values(["precision", "recall", "wait"], ascending=[False, False, True]).iloc[0]
            met_target = False
        else:
            selected = eligible.sort_values(["recall", "wait"], ascending=[False, True]).iloc[0]
            met_target = True
        test = evaluation_summary[
            (evaluation_summary.dataset == dataset)
            & (evaluation_summary.state_kind == state_kind)
            & (evaluation_summary.wait == int(selected.wait))
        ].iloc[0]
        immediate = evaluation_summary[
            (evaluation_summary.dataset == dataset)
            & (evaluation_summary.state_kind == state_kind)
            & (evaluation_summary.wait == 1)
        ].iloc[0]
        selections.append({
            "dataset": dataset,
            "state_kind": state_kind,
            "selected_wait": int(selected.wait),
            "calibration_precision": float(selected.precision),
            "calibration_recall": float(selected.recall),
            "met_target": bool(met_target),
            "evaluation_precision": float(test.precision),
            "evaluation_recall": float(test.recall),
            "evaluation_acceptance_rate": float(test.acceptance_rate),
            "immediate_evaluation_precision": float(immediate.precision),
            "precision_delta": float(test.precision - immediate.precision),
        })
    selected = pd.DataFrame(selections)
    selected.to_csv(args.output / "selected_certifier_evaluation.csv", index=False)

    lines = [
        "# Transition certifier: calibration/evaluation split",
        "",
        "The smallest/highest-recall wait meeting 0.95 precision is selected on the first seed; two other seeds are evaluation only.",
        "",
        "| Dataset | State | Selected wait | Calibration P/R | Evaluation P/R | Immediate P | Delta P | Acceptance |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.state_kind} | {row.selected_wait} | "
            f"{row.calibration_precision:.4f}/{row.calibration_recall:.4f} | "
            f"{row.evaluation_precision:.4f}/{row.evaluation_recall:.4f} | "
            f"{row.immediate_evaluation_precision:.4f} | {row.precision_delta:+.4f} | "
            f"{row.evaluation_acceptance_rate:.4f} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "source_job_id": args.job_id,
        "input_files": len(paths),
        "target_precision": args.target_precision,
        "selection_seed_policy": "minimum seed per dataset for calibration; remaining two seeds for evaluation",
        "api_calls": 0,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
