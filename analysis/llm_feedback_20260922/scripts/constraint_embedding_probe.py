#!/usr/bin/env python3
"""Probe whether frozen FM embeddings linearly expose condition/row compatibility.

This is deliberately an offline representation experiment, not an online LLM
controller.  Only synthetic records are embedded.  IDs and benchmark labels are
never included in API text.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from llm_feedback_audit import load_env
from semantic_llm_sweep import api_json


def render(case: dict, representation: str) -> str:
    row = {name: case["record"].get(name) for name in case["involved_columns"]}
    if representation == "natural_language_pair":
        return "Constraint: " + case["constraint_description"] + "\nRecord fields: " + json.dumps(row, sort_keys=True)
    if representation == "structured_pair":
        return "Predicate: " + json.dumps(case["structured_predicate"], sort_keys=True) + "\nRecord fields: " + json.dumps(row, sort_keys=True)
    if representation == "condition_only":
        return "Constraint: " + case["constraint_description"]
    if representation == "row_only":
        return "Record fields: " + json.dumps(row, sort_keys=True)
    raise ValueError(representation)


def embeddings(key: str, model: str, texts: list[str], batch_size: int) -> tuple[np.ndarray, dict]:
    chunks, usage = [], {"prompt_tokens": 0, "total_tokens": 0, "calls": 0, "latency_seconds": 0.0}
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        error = None
        for attempt in range(5):
            try:
                began = time.perf_counter()
                response = api_json(
                    "https://api.openai.com/v1/embeddings",
                    key,
                    {"model": model, "input": batch, "encoding_format": "float"},
                    timeout=240,
                )
                usage["latency_seconds"] += time.perf_counter() - began
                usage["calls"] += 1
                usage["prompt_tokens"] += int(response.get("usage", {}).get("prompt_tokens", 0) or 0)
                usage["total_tokens"] += int(response.get("usage", {}).get("total_tokens", 0) or 0)
                ordered = sorted(response["data"], key=lambda item: item["index"])
                chunks.append(np.asarray([item["embedding"] for item in ordered], dtype=np.float32))
                error = None
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                error = RuntimeError(f"HTTP {exc.code}: {body}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
                time.sleep(min(30.0, 2**attempt + random.random()))
            except Exception as exc:  # transient network/JSON errors
                error = exc
                time.sleep(min(30.0, 2**attempt + random.random()))
        if error is not None:
            raise error
    return np.concatenate(chunks), usage


def evaluate_features(x, frame: pd.DataFrame, feature_name: str, sparse_text: bool = False) -> list[dict]:
    rows = []
    targets = {
        "applicability": frame.gold_applicability.to_numpy(),
        "violation": np.where(frame.gold_status.eq("violated"), "violated", "not_violated"),
        "status": frame.gold_status.to_numpy(),
    }
    for split_name, groups in {
        "leave_constraint_out": frame.constraint_id,
        "leave_timestep_out": frame.t.map(lambda value: f"{float(value):.6f}"),
    }.items():
        for held_out in sorted(groups.unique()):
            test = groups.eq(held_out).to_numpy()
            train = ~test
            for task, y in targets.items():
                if sparse_text:
                    estimator = make_pipeline(
                        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000),
                        LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0),
                    )
                else:
                    estimator = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
                estimator.fit([x[i] for i in np.flatnonzero(train)] if sparse_text else x[train], y[train])
                pred = estimator.predict([x[i] for i in np.flatnonzero(test)] if sparse_text else x[test])
                rows.append({
                    "feature": feature_name,
                    "split": split_name,
                    "held_out": held_out,
                    "task": task,
                    "train_n": int(train.sum()),
                    "test_n": int(test.sum()),
                    "accuracy": accuracy_score(y[test], pred),
                    "macro_f1": f1_score(y[test], pred, average="macro", zero_division=0),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--representation", choices=["natural_language_pair", "structured_pair", "condition_only", "row_only"], required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    texts = [render(case, args.representation) for case in cases]
    frame = pd.DataFrame({
        "constraint_id": [case["constraint_id"] for case in cases],
        "t": [case["t"] for case in cases],
        "gold_applicability": [case["gold_applicability"] for case in cases],
        "gold_status": [case["gold_status"] for case in cases],
    })
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")
    x, usage = embeddings(key, args.model, texts, args.batch_size)
    np.save(args.output / "embeddings.npy", x)
    # Local files contain synthetic text only and allow an exact audit of what crossed the boundary.
    with (args.output / "api_inputs.jsonl").open("w") as handle:
        for text_value in texts:
            handle.write(json.dumps({"text": text_value}) + "\n")
    result_rows = evaluate_features(x, frame, f"embedding:{args.model}")
    result_rows += evaluate_features(texts, frame, "tfidf", sparse_text=True)
    metrics = pd.DataFrame(result_rows)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    aggregate = metrics.groupby(["feature", "split", "task"], as_index=False).agg(
        folds=("held_out", "count"), accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    )
    aggregate.to_csv(args.output / "aggregate_metrics.csv", index=False)
    summary = {
        "status": "executed",
        "model": args.model,
        "representation": args.representation,
        "cases": len(cases),
        "embedding_dimension": int(x.shape[1]),
        **usage,
        "privacy_boundary": "Only synthetic relevant-column snippets and constraint text; no real rows or benchmark labels sent.",
        "interpretation": "Linear-probe feasibility only; not evidence of end-to-end conditioned generation.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
