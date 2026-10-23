#!/usr/bin/env python3
"""Compile ConstrainedDGM's five public News inequalities into typed edges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RELATIONS = [
    ("tokens_title_le_content", " n_tokens_title", " n_tokens_content"),
    ("nonstop_unique_le_nonstop_words", " n_non_stop_unique_tokens", " n_non_stop_words"),
    ("self_hrefs_le_hrefs", " num_self_hrefs", " num_hrefs"),
    ("selfref_min_le_max", " self_reference_min_shares", " self_reference_max_shares"),
    ("selfref_avg_le_max", " self_reference_avg_sharess", " self_reference_max_shares"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(args.repo / "data/news/train.csv")
    graph = []
    for index, (label, left, right) in enumerate(RELATIONS):
        valid = pd.to_numeric(train[left], errors="coerce").le(
            pd.to_numeric(train[right], errors="coerce") + args.tolerance
        )
        violations = int((~valid).sum())
        rate = violations / len(train)
        atom = {
            "left_column": left,
            "operator": "<=",
            "right_kind": "column",
            "right_column": right,
            "right_value": 0,
            "tolerance": args.tolerance,
        }
        graph.append({
            "candidate_id": f"news_public_{index:03d}",
            "description": f"{left.strip()} <= {right.strip()}",
            "applicability": [],
            "requirements": [atom],
            "exceptions": [],
            "relation_family": "external_linear_inequality",
            "hardness_prior": "hard",
            "classification": "hard",
            "source": "constrained_dgm_iclr2024_public_constraints",
            "evidence_basis": "external_specification_verified_on_train",
            "provenance": ["ConstrainedDGM ICLR 2024", "train_only_verification"],
            "usage_role": "actuate",
            "confidence": 1.0 - rate,
            "calibration": {
                "support": len(train),
                "violations": violations,
                "violation_rate": rate,
            },
            "public_label": label,
        })
    (args.output / "graph.json").write_text(json.dumps(graph, indent=2))
    metadata = {
        "dataset": "news",
        "source": "ConstrainedDGM ICLR 2024 public constraint file",
        "edges": len(graph),
        "train_rows": len(train),
        "heldout_used": False,
        "tolerance": args.tolerance,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
