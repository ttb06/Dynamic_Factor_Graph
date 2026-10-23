#!/usr/bin/env python3
"""Embed researcher-written mixed-condition strings; never sends table rows."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from llm_feedback_audit import load_env
from mixed_type_condition_adapter import make_catalog


def render(dataset: str, contract: dict, factorized: bool) -> str:
    clauses = []
    for column, operator, value in contract["predicates"]:
        op = ">=" if operator == "ge" else "=="
        rendered_value = "<typed parameter supplied separately>" if factorized else json.dumps(value)
        clauses.append(f"column {column} {op} {rendered_value}")
    return (
        "Synthetic tabular cohort condition. "
        f"Schema/domain: {dataset}. Required predicates: " + " AND ".join(clauses) + "."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["adult", "default", "shoppers", "diabetes"], required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="text-embedding-3-large")
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--factorized", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")
    contracts = make_catalog(args.dataset)
    texts = [render(args.dataset, contract, args.factorized) for contract in contracts]
    unique = list(dict.fromkeys(texts))
    request = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps({"model": args.model, "input": unique, "dimensions": args.dimensions}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = json.loads(response.read().decode())
    ordered = sorted(payload["data"], key=lambda row: row["index"])
    by_text = {text: row["embedding"] for text, row in zip(unique, ordered)}
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "dimensions": args.dimensions,
        "factorized": args.factorized,
        "vectors": {contract["contract_id"]: by_text[text] for contract, text in zip(contracts, texts)},
        "usage": payload.get("usage", {}),
        "latency_seconds": time.perf_counter() - started,
        "privacy": "Only researcher-written schema/condition strings; no real or synthetic rows.",
    }
    (args.output / "embeddings.json").write_text(json.dumps(result))
    (args.output / "api_inputs.json").write_text(json.dumps(unique, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "vectors"}, indent=2))


if __name__ == "__main__":
    main()
