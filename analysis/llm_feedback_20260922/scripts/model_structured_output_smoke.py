#!/usr/bin/env python3
"""One-call smoke test for model access and the graph protocol JSON schema."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from knowledge_graph_protocol import PROPOSAL_SCHEMA, native, retry_call
from llm_feedback_audit import load_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reasoning-effort", default="high")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_env(args.env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable")
    payload = {
        "privacy": "synthetic_schema_only",
        "dataset_context": "online session with page counts and durations",
        "schema": [
            {"name": "page_count", "type": "numeric", "minimum": 0},
            {"name": "page_duration", "type": "numeric", "minimum": 0},
        ],
        "request": "Return exactly one directional typed edge. Use an empty exception list if none is needed.",
    }
    result, usage, latency, error = retry_call(
        key, args.model,
        "Generate one executable non-tautological knowledge edge. If page_count is zero, page_duration should be zero.",
        payload, PROPOSAL_SCHEMA, "graph_protocol_smoke", args.reasoning_effort, maximum=5000,
    )
    artifact = {"model": args.model, "result": result, "usage": usage, "latency": latency, "error": error}
    (args.output / "smoke.json").write_text(json.dumps(native(artifact), indent=2))
    if error:
        raise RuntimeError(error)
    if len(result.get("candidates", [])) != 1:
        raise RuntimeError(f"expected one candidate, received {len(result.get('candidates', []))}")
    print(json.dumps(native(artifact), indent=2))


if __name__ == "__main__":
    main()
