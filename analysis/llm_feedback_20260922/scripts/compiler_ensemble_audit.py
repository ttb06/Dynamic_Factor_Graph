#!/usr/bin/env python3
"""Self-consistency and abstention audit over repeated compiler outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def evaluate_group(name: str, runs: list[tuple[Path, dict]], rows: list[dict]) -> None:
    candidates = []
    total_tokens = 0
    for path, summary in runs:
        frame = pd.read_csv(path / "decisions.csv")[["case_id", "gold_status", "pred_status"]]
        frame = frame.rename(columns={"pred_status": f"pred_{len(candidates)}"})
        candidates.append(frame)
        total_tokens += int(summary.get("total_tokens", 0) or 0)
    merged = candidates[0]
    for frame in candidates[1:]:
        merged = merged.merge(frame.drop(columns=["gold_status"]), on="case_id", how="inner")
    pred_cols = [column for column in merged if column.startswith("pred_")]
    for threshold in (0.5, 2 / 3, 0.8, 1.0):
        covered = correct = 0
        confidences = []
        for row in merged.itertuples(index=False):
            values = [getattr(row, column) for column in pred_cols if getattr(row, column) != "missing"]
            if not values:
                continue
            counts = Counter(values)
            label, votes = counts.most_common(1)[0]
            confidence = votes / len(values)
            # Strict majority prevents arbitrary decisions under ties.
            accepted = votes > len(values) / 2 and confidence + 1e-12 >= threshold
            if accepted:
                covered += 1
                correct += int(label == row.gold_status)
                confidences.append(confidence)
        rows.append({
            "ensemble":name, "members":len(runs), "threshold":threshold,
            "cases":len(merged), "coverage":covered / len(merged),
            "selective_accuracy":correct / covered if covered else None,
            "end_to_end_correct_coverage":correct / len(merged),
            "mean_accepted_agreement":sum(confidences) / len(confidences) if confidences else None,
            "tokens":total_tokens,
        })


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    runs=[]
    for path in sorted(args.run_root.glob("request_condition_api_expanded_*_compile_*")):
        summary_path=path/"summary.json"
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            if summary.get("compile_strategy", "basic") == "basic":
                runs.append((path,summary))
    rows=[]
    for model in sorted({summary["model"] for _,summary in runs}):
        evaluate_group(model,[(path,summary) for path,summary in runs if summary["model"]==model],rows)
    evaluate_group("cross_model",runs,rows)
    result=pd.DataFrame(rows); result.to_csv(args.output/"ensemble_frontier.csv",index=False)
    lines=["# Compiler self-consistency and abstention","","No gold label is used for voting. Gold is used only after the consensus decision to report accuracy.","",
           "| ensemble | members | agreement threshold | coverage | selective accuracy | correct coverage | tokens |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result.itertuples(index=False):
        lines.append(f"| {row.ensemble} | {row.members} | {row.threshold:.3f} | {row.coverage:.3f} | {row.selective_accuracy:.3f} | {row.end_to_end_correct_coverage:.3f} | {row.tokens} |")
    lines += ["","This is an offline uncertainty diagnostic. Repeated compilation increases token cost, and agreement can be confidently wrong; independent semantic validation remains necessary."]
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print(result.to_string(index=False))


if __name__=="__main__": main()
