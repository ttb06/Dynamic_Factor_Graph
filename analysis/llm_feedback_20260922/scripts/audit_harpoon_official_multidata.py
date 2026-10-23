#!/usr/bin/env python3
"""Audit official Harpoon reruns separately from paper-reported artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def md_table(frame: pd.DataFrame) -> str:
    def fmt(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(fmt(value) for value in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[column for column in frame.columns if column.startswith("Unnamed:")], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    reruns = []
    missing = []
    for dataset in ["adult", "default", "shoppers"]:
        path = args.run_root / dataset / "experiments" / "general_constraints_updated_utility.csv"
        if not path.exists():
            missing.append(dataset)
            continue
        frame = clean(pd.read_csv(path))
        frame["evidence"] = "official_code_rerun_1_trial"
        reruns.append(frame)
    rerun = pd.concat(reruns, ignore_index=True) if reruns else pd.DataFrame()
    rerun.to_csv(args.output / "official_rerun_rows.csv", index=False)

    paper = clean(pd.read_csv(args.paper_csv))
    paper["evidence"] = "paper_repository_reported_5_trials"
    paper.to_csv(args.output / "paper_reported_rows.csv", index=False)

    columns = ["Dataset", "Method", "Constraint", "Avg ViolationAcc", "Avg Alpha-P", "Avg ks",
               "Avg detect", "Avg identifiability", "Avg xgb utility", "evidence"]
    comparison = pd.concat([paper[columns], rerun[columns] if not rerun.empty else pd.DataFrame(columns=columns)],
                           ignore_index=True)
    comparison.to_csv(args.output / "paper_and_rerun_long.csv", index=False)
    lines = [
        "# Official Harpoon multi-dataset audit",
        "",
        "Paper rows and local reruns are intentionally kept separate. The local batch uses one trial as a compatibility run; the paper reports five trials.",
        "",
        md_table(comparison),
        "",
        f"Missing rerun CSVs: `{json.dumps(missing)}`.",
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))
    (args.output / "run_summary.json").write_text(json.dumps({
        "rerun_rows": len(rerun), "paper_rows": len(paper), "missing_datasets": missing,
        "comparison_policy": "never rank paper-reported and local-rerun rows as a matched protocol",
    }, indent=2))


if __name__ == "__main__":
    main()
