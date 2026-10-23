#!/usr/bin/env python3
"""Aggregate request-conditioned API probe results after Slurm jobs finish."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small frame without pandas' optional tabulate dependency."""

    def value(item) -> str:
        if pd.isna(item):
            return ""
        if isinstance(item, float):
            return f"{item:.4g}"
        return str(item).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value(item) for item in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-ids", required=True, help="comma- or colon-separated Slurm job IDs")
    parser.add_argument("--local-summary", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    wanted = {item.strip() for item in re.split(r"[:,]", args.job_ids) if item.strip()}
    run_dirs = sorted(
        path for path in args.run_root.glob("request_condition_api_*")
        if path.is_dir() and any(path.name.endswith(f"_{job_id}") for job_id in wanted)
    )
    summaries, frames = [], []
    for path in run_dirs:
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        summary["run_dir"] = str(path)
        summary["job_id"] = next((job_id for job_id in wanted if path.name.endswith(f"_{job_id}")), "")
        summaries.append(summary)
        decision_path = path / "decisions.csv"
        if decision_path.exists():
            frame = pd.read_csv(decision_path)
            frame["model"] = summary.get("model")
            frame["mode"] = summary.get("mode")
            frame["job_id"] = summary["job_id"]
            frames.append(frame)
    pd.DataFrame(summaries).to_csv(args.output / "runs.csv", index=False)
    if frames:
        decisions = pd.concat(frames, ignore_index=True)
        decisions.to_csv(args.output / "decisions.csv", index=False)
        by_design = (
            decisions.assign(correct=lambda x: x.gold_status.eq(x.pred_status))
            .groupby(["model", "mode", "design"], dropna=False)
            .agg(cases=("case_id", "size"), status_accuracy=("correct", "mean"))
            .reset_index()
        )
        by_design.to_csv(args.output / "by_design.csv", index=False)
        by_status = (
            decisions.assign(correct=lambda x: x.gold_status.eq(x.pred_status))
            .groupby(["model", "mode", "gold_status"], dropna=False)
            .agg(cases=("case_id", "size"), status_accuracy=("correct", "mean"))
            .reset_index()
        )
        by_status.to_csv(args.output / "by_status.csv", index=False)
    else:
        by_design = pd.DataFrame()
        by_status = pd.DataFrame()

    local = None
    if args.local_summary and args.local_summary.exists():
        local = json.loads(args.local_summary.read_text())
        (args.output / "local_baselines.json").write_text(json.dumps(local, indent=2))
    lines = [
        "# Request-conditioned constraint probe",
        "",
        "This is an executed synthetic-only pilot. API inputs used opaque randomized IDs; oracle DSLs, gold status, pair IDs and design labels stayed local.",
        "",
        "## API runs",
        "",
    ]
    if summaries:
        cols = [
            "model", "mode", "four_way_status_accuracy", "request_swap_pair_accuracy",
            "exception_false_action_rate", "execution_equivalence", "dsl_structural_exact_match_rate",
            "total_tokens", "latency_seconds", "status",
        ]
        table = pd.DataFrame(summaries)
        cols = [column for column in cols if column in table]
        lines.append(markdown_table(table[cols]))
    else:
        lines.append("No completed API run was found (check Slurm logs and job IDs).")
    lines.extend(["", "## Local controls", ""])
    if local:
        for name, result in local.get("baselines", {}).items():
            lines.append(
                f"- {name}: status accuracy={result.get('four_way_status_accuracy')}, "
                f"swap-pair accuracy={result.get('request_swap_pair_accuracy')}, "
                f"exception false-action={result.get('exception_false_action_rate')}."
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- A perfect deterministic template parser is expected on this controlled six-request grammar; it is evidence that an LLM is unnecessary for templated constraints, not evidence of general semantic parsing.",
            "- Direct judgement and compile-then-execute are compared on identical synthetic rows and requests. Compile execution equivalence isolates representation errors from executor errors.",
            "- The key anti-shortcut metric is exact accuracy over same-row/different-request swap pairs; marginal row plausibility cannot solve those pairs.",
            "- This pilot does not establish table/domain generalization or fidelity effects on generated distributions.",
        ]
    )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    metadata = {"status": "executed", "requested_jobs": sorted(wanted), "loaded_runs": len(summaries), "run_dirs": [str(path) for path in run_dirs]}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
