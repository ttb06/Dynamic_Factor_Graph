#!/usr/bin/env python3
"""Aggregate per-timestep applicability gate audits across seeds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ORDER = ["(0.75,1.0]", "(0.50,0.75]", "(0.25,0.50]", "(0.10,0.25]", "[0,0.10]"]


def markdown(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    wanted = {token for token in re.split(r"[:,]", args.job_ids) if token}
    frames, metadata = [], []
    for path in args.run_root.glob("applicability_gate_trajectory_*"):
        if not any(path.name.endswith(f"_{job_id}") for job_id in wanted):
            continue
        frame = pd.read_csv(path / "gate_trajectory.csv")
        frame["run_dir"] = str(path)
        frames.append(frame)
        metadata.append(json.loads((path / "run_metadata.json").read_text()))
    if not frames:
        raise RuntimeError("no trajectory outputs found")
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(args.output / "all_gate_trajectory.csv", index=False)
    all_rows["time_bin"] = pd.Categorical(all_rows["time_bin"], ORDER, ordered=True)
    by_seed = all_rows.groupby(["seed", "guidance_scale", "split", "time_bin"], observed=True).agg(
        contracts=("contract_id", "nunique"), gate_accuracy=("gate_accuracy_to_final", "mean"),
        gate_precision=("gate_precision_to_final", "mean"), gate_recall=("gate_recall_to_final", "mean"),
        flip_rate=("flip_rate_from_previous", "mean"),
        relevant_category_agreement=("relevant_category_agreement_to_final", "mean"),
        all_category_agreement=("all_category_agreement_to_final", "mean"),
    ).reset_index()
    by_seed.to_csv(args.output / "summary_by_seed.csv", index=False)
    stability = by_seed.groupby(["guidance_scale", "split", "time_bin"], observed=True).agg(
        seeds=("seed", "nunique"), gate_accuracy=("gate_accuracy", "mean"), gate_accuracy_std=("gate_accuracy", "std"),
        gate_precision=("gate_precision", "mean"), gate_recall=("gate_recall", "mean"),
        flip_rate=("flip_rate", "mean"), relevant_category_agreement=("relevant_category_agreement", "mean"),
        all_category_agreement=("all_category_agreement", "mean"),
    ).reset_index().sort_values(["guidance_scale", "split", "time_bin"])
    stability.to_csv(args.output / "stability.csv", index=False)
    concise = stability[stability["guidance_scale"].eq(2.0)][[
        "split", "time_bin", "seeds", "gate_accuracy", "gate_accuracy_std", "gate_precision",
        "gate_recall", "flip_rate", "relevant_category_agreement",
    ]]
    text = [
        "# Applicability gate trajectory audit",
        "",
        "The gate uses the unconditional categorical preview at every denoiser call. Metrics compare that decision with applicability of the final generated row; this is a temporal consistency diagnostic, not ground-truth semantic accuracy.",
        "",
        "## CFG 2 across seeds",
        "",
        markdown(concise),
        "",
        "A useful intervention window requires both high agreement with the final row and low flip rate. High apparent accuracy on a rare antecedent can be trivial negative prediction, so precision/recall are reported alongside accuracy.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(text) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({"job_ids": sorted(wanted), "runs": metadata}, indent=2))
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    main()
