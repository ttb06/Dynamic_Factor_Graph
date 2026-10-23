#!/usr/bin/env python3
"""Aggregate isolated five-trial runs of the official Harpoon repository."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(args.root.glob(f"harpoon_official_5trial_*_{args.source_job}_*/general_constraints_updated_utility.csv"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected 3 dataset files for job {args.source_job}, found {len(paths)}: {paths}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(args.output / "official_harpoon_5trial.csv", index=False)
    columns = ["Dataset", "Constraint", "Avg ViolationAcc", "Std ViolationAcc", "Avg Alpha-P", "Avg ks", "Avg detect", "Avg xgb utility"]
    columns = [column for column in columns if column in result]
    lines = [
        "# Official Harpoon five-trial rerun", "",
        "This table was produced by the authors' public code, pretrained checkpoints,",
        "200 reverse steps, update strength 0.2, and five trials.", "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in result[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row) + " |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_metadata.json").write_text(json.dumps({
        "source_array_job": args.source_job,
        "scope": "official Harpoon code and checkpoints; independent backbone/protocol",
        "datasets": sorted(result["Dataset"].astype(str).unique().tolist()),
        "trials": 5,
        "reverse_steps": 200,
        "guidance_strength": 0.2,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
