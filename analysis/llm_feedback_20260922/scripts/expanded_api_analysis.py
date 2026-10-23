#!/usr/bin/env python3
"""Failure and cost analysis for the expanded request-condition API runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    requests = pd.DataFrame([json.loads(line) for line in (args.bank / "requests.jsonl").read_text().splitlines() if line])
    request_meta = requests[["request_id", "family", "surface_id"]]
    run_rows, decision_frames, compile_frames = [], [], []
    paths = sorted(args.run_root.glob("request_condition_api_expanded_*"))
    for path in paths:
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        batch = np.nan
        match = re.search(r"_direct_b(\d+)_", path.name)
        if match:
            batch = int(match.group(1))
        seed_match = re.search(r"_s(\d+)_\d+$", path.name)
        seed = int(seed_match.group(1)) if seed_match else -1
        overall = summary.get("execution_equivalence") if summary["mode"] == "compile_execute" else summary.get("four_way_status_accuracy")
        run_rows.append({
            "run_dir": str(path), "model":summary["model"], "mode":summary["mode"],
            "batch_size":batch, "seed":seed, "overall_accuracy":overall,
            "action_accuracy":summary.get("action_accuracy"), "exception_false_action":summary.get("exception_false_action_rate"),
            "not_applicable_false_action":summary.get("not_applicable_false_action_rate"),
            "missing_rate":summary.get("missing_rate"), "total_tokens":summary.get("total_tokens"),
            "latency_seconds":summary.get("latency_seconds"), "structural_exact":summary.get("dsl_structural_exact_match_rate"),
        })
        decisions = pd.read_csv(path / "decisions.csv").merge(request_meta, on="request_id", how="left")
        decisions["correct"] = decisions.gold_status.eq(decisions.pred_status)
        decisions["model"] = summary["model"]; decisions["mode"] = summary["mode"]
        decisions["batch_size"] = batch; decisions["seed"] = seed; decisions["run_dir"] = str(path)
        decision_frames.append(decisions)
        compile_path = path / "compiled_requests.csv"
        if compile_path.exists():
            compiled = pd.read_csv(compile_path).merge(request_meta, on="request_id", how="left")
            compiled["model"] = summary["model"]; compiled["seed"] = seed; compiled["run_dir"] = str(path)
            compile_frames.append(compiled)

    runs = pd.DataFrame(run_rows)
    decisions = pd.concat(decision_frames, ignore_index=True)
    compiled = pd.concat(compile_frames, ignore_index=True)
    runs.to_csv(args.output / "runs.csv", index=False)
    decisions.to_csv(args.output / "decisions.csv", index=False)
    compiled.to_csv(args.output / "compiled_requests.csv", index=False)
    summary = runs.groupby(["model", "mode", "batch_size"], dropna=False).agg(
        runs=("seed", "size"), accuracy=("overall_accuracy", "mean"), accuracy_min=("overall_accuracy", "min"),
        accuracy_max=("overall_accuracy", "max"), action_accuracy=("action_accuracy", "mean"),
        exception_false_action=("exception_false_action", "mean"), missing_rate=("missing_rate", "mean"),
        tokens=("total_tokens", "mean"), latency=("latency_seconds", "mean"), structural_exact=("structural_exact", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "run_summary.csv", index=False)
    by_surface = decisions.groupby(["model", "mode", "batch_size", "surface_id"], dropna=False).agg(
        cases=("case_id", "size"), accuracy=("correct", "mean")
    ).reset_index()
    by_surface.to_csv(args.output / "by_surface.csv", index=False)
    by_status = decisions.groupby(["model", "mode", "batch_size", "gold_status"], dropna=False).agg(
        cases=("case_id", "size"), accuracy=("correct", "mean")
    ).reset_index()
    by_status.to_csv(args.output / "by_status.csv", index=False)
    by_family = decisions.groupby(["model", "mode", "batch_size", "family"], dropna=False).agg(
        cases=("case_id", "size"), accuracy=("correct", "mean")
    ).reset_index()
    by_family.to_csv(args.output / "by_family.csv", index=False)
    compile_stability = decisions[decisions["mode"].eq("compile_execute")].groupby(["model", "case_id"]).agg(
        seed_agreement=("pred_status", lambda values: float(pd.Series(values).nunique() == 1)),
        all_correct=("correct", "all"), any_correct=("correct", "any")
    ).reset_index().groupby("model").agg(
        cases=("case_id", "size"), seed_agreement=("seed_agreement", "mean"),
        all_seeds_correct=("all_correct", "mean"), any_seed_correct=("any_correct", "mean")
    ).reset_index()
    compile_stability.to_csv(args.output / "compile_stability.csv", index=False)

    lines = [
        "# Expanded natural-language contract audit", "",
        "Executed on 50 requests (10 semantics × 5 surface forms) and 400 synthetic-only cases, balanced over satisfied, violated, not-applicable and excepted.", "",
        "| model | mode | batch | runs | accuracy | range | action accuracy | exception false-action | missing | tokens | latency (s) | structural exact |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        batch = "compile-once" if pd.isna(row.batch_size) else str(int(row.batch_size))
        structural = "" if pd.isna(row.structural_exact) else f"{row.structural_exact:.3f}"
        lines.append(
            f"| {row.model} | {row.mode} | {batch} | {row.runs} | {row.accuracy:.3f} | "
            f"[{row.accuracy_min:.3f}, {row.accuracy_max:.3f}] | {row.action_accuracy:.3f} | "
            f"{row.exception_false_action:.3f} | {row.missing_rate:.3f} | {row.tokens:.0f} | {row.latency:.1f} | {structural} |"
        )
    lines += ["", "## Compile stability", "", "| model | cases | identical status across seeds | correct in all seeds | correct in at least one seed |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in compile_stability.itertuples(index=False):
        lines.append(f"| {row.model} | {row.cases} | {row.seed_agreement:.3f} | {row.all_seeds_correct:.3f} | {row.any_seed_correct:.3f} |")
    lines += [
        "", "## Interpretation", "",
        "- Compile-once is substantially more accurate and roughly an order of magnitude cheaper in tokens than repeated row judgement in this benchmark.",
        "- Compile variance is material: a valid JSON schema does not guarantee semantic equivalence. Ensemble/validation/abstention remains necessary.",
        "- Smaller direct batches help, but still spend more tokens because request and row context are repeatedly transmitted.",
        "- This is controlled language over one schema. It does not establish robustness to arbitrary policy documents, unseen operators, cross-row constraints or domain ontology.",
        "- No real row was sent to the API; the benchmark cannot measure privacy leakage from real records.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"loaded {len(runs)} runs -> {args.output}")


if __name__ == "__main__":
    main()
