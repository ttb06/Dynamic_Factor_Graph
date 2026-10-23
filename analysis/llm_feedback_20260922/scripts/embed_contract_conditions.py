#!/usr/bin/env python3
"""Embed public synthetic-generation contract text with an OpenAI embedding model."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from contract_condition_catalog import catalog
from llm_feedback_audit import load_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="text-embedding-3-large")
    parser.add_argument("--dimensions", type=int, default=512)
    parser.add_argument("--factorized", action="store_true",
                        help="Remove numeric values from text; typed AST supplies all parameters.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")

    contracts = catalog()
    if args.factorized:
        inputs = []
        for item in contracts:
            fields = ", ".join(predicate["column"] for predicate in item["predicates"])
            inputs.append(
                "Mixed-type tabular generation contract. Apply lower-bound requirements to these fields: "
                f"{fields}. Exact operators and numeric parameters are supplied in a separate typed channel."
            )
    else:
        inputs = [item["text"] for item in contracts]
    unique_inputs = list(dict.fromkeys(inputs))
    payload = json.dumps(
        {"model": args.model, "input": unique_inputs, "dimensions": args.dimensions}
    ).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.loads(response.read().decode())
    ordered = sorted(result["data"], key=lambda row: row["index"])
    by_text = {text: row["embedding"] for text, row in zip(unique_inputs, ordered)}
    vectors = {contract["contract_id"]: by_text[text] for contract, text in zip(contracts, inputs)}
    output = {
        "model": args.model,
        "dimensions": args.dimensions,
        "vectors": vectors,
        "usage": result.get("usage", {}),
        "latency_seconds": time.perf_counter() - started,
        "factorized": args.factorized,
        "unique_embedding_inputs": len(unique_inputs),
        "privacy": "Only researcher-written contract strings were sent; no real or synthetic table rows.",
    }
    (args.output / "embeddings.json").write_text(json.dumps(output))
    (args.output / "run_metadata.json").write_text(
        json.dumps({key: value for key, value in output.items() if key != "vectors"} | {"contracts": len(contracts)}, indent=2)
    )
    print(json.dumps({key: value for key, value in output.items() if key != "vectors"} | {"contracts": len(contracts)}, indent=2))


if __name__ == "__main__":
    main()
