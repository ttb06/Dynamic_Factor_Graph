#!/usr/bin/env python3
"""Aggregate controlled LLM-gated rollback jobs without assuming every job succeeded."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    requested = [item.strip() for item in args.job_ids.split(",") if item.strip()]
    metrics_frames: list[pd.DataFrame] = []
    detection_rows: list[dict] = []
    loaded: list[str] = []
    missing: list[str] = []

    for job_id in requested:
        matches = sorted(args.root.glob(f"llm_rollback_*_{job_id}"))
        if not matches or not (matches[0] / "rollback_metrics.csv").exists():
            missing.append(job_id)
            continue
        directory = matches[0]
        metadata = json.loads((directory / "run_metadata.json").read_text())
        condition = str(metadata["condition"])
        target_t = float(metadata["target_t"])
        model = str(metadata["model"])
        experiment = directory.name.removeprefix("llm_rollback_").removesuffix(f"_{job_id}")

        frame = pd.read_csv(directory / "rollback_metrics.csv")
        frame.insert(0, "source_job", job_id)
        frame.insert(1, "experiment", experiment)
        frame.insert(2, "condition", condition)
        frame.insert(3, "model", model)
        frame.insert(4, "target_t", target_t)
        metrics_frames.append(frame)

        row_decisions = []
        detail_path = directory / "llm_details.jsonl"
        if detail_path.exists():
            for line in detail_path.read_text().splitlines():
                item = json.loads(line)
                if "row" in item:
                    row_decisions.append(item)
        tp = sum(item.get("predicted_relation") == item.get("true_injected_relation") for item in row_decisions)
        wrong = sum(
            bool(item.get("predicted_relation"))
            and item.get("predicted_relation") != item.get("true_injected_relation")
            for item in row_decisions
        )
        missed = sum(not item.get("predicted_relation") for item in row_decisions)
        usage = metadata.get("branch_metadata", {}).get("llm_rollback", {}).get("usage", {})
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        detection_rows.append(
            {
                "source_job": job_id,
                "experiment": experiment,
                "condition": condition,
                "model": model,
                "target_t": target_t,
                "records": len(row_decisions),
                "exact_relation_tp": tp,
                "wrong_relation": wrong,
                "missed": missed,
                "exact_relation_recall": safe_div(tp, len(row_decisions)),
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "total_tokens": total_tokens,
                "tp_per_1k_tokens": safe_div(1000 * tp, total_tokens),
                "api_latency_seconds": metadata.get("branch_metadata", {})
                .get("llm_rollback", {})
                .get("api_latency_seconds", 0),
            }
        )
        loaded.append(job_id)

    if not metrics_frames:
        raise RuntimeError("No completed rollback outputs found")

    combined = pd.concat(metrics_frames, ignore_index=True)
    combined.to_csv(args.output / "combined_rollback_metrics.csv", index=False)
    detection = pd.DataFrame(detection_rows)
    detection.to_csv(args.output / "llm_detection_and_cost.csv", index=False)

    piv = combined.pivot_table(
        index=["source_job", "experiment", "condition", "model", "target_t", "corruption_type"],
        columns="branch",
        values="final_intended_violation_rate",
    ).reset_index()
    for baseline in ("corrupt_no_feedback", "static_verifier_rollback", "oracle_rollback"):
        if baseline in piv and "llm_rollback" in piv:
            piv[f"llm_reduction_vs_{baseline}"] = piv[baseline] - piv["llm_rollback"]
    piv.to_csv(args.output / "branch_comparison.csv", index=False)

    summary = {
        "status": "executed",
        "scope": "controlled synthetic latent-corruption rollback upper-bound stress test",
        "jobs_requested": requested,
        "jobs_loaded": loaded,
        "missing_jobs": missing,
        "privacy_boundary": "Only synthetic previews were sent to the API; no real rows.",
        "interpretation_warning": (
            "Rollback restores pre-corruption latent columns, so these results test critic timing/gating; "
            "they are not a deployment-ready editing result."
        ),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(summary, indent=2))
    print(detection.to_string(index=False))
    print("\nBranch comparison:\n", piv.to_string(index=False))


if __name__ == "__main__":
    main()
