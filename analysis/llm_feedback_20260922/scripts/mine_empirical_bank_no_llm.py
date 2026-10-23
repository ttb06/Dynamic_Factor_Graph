#!/usr/bin/env python3
"""Mine and validate a constraint bank from the training file only.

This is the API-free extraction of the empirical branch that previously lived
inside the mixed LLM protocol.  It never opens the repository test file and
does not import API keys or call a model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--maximum", type=int, default=30)
    parser.add_argument("--exclude-target", action="store_true",
                        help="Exclude candidates that read or modify the prediction target.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    scripts = args.repo / "analysis/llm_feedback_20260922/scripts"
    sys.path.insert(0, str(scripts))
    import cross_dataset_constraint_audit as audit
    from knowledge_graph_protocol import (
        CANDIDATE_REQUIRED, canonical, classify_evidence, empirical_hints, verify,
    )

    data = args.repo / "data" / args.dataset
    info = json.loads((data / "info.json").read_text())
    train = pd.read_csv(data / "train.csv")
    columns = list(train.columns)
    numerical = [columns[int(index)] for index in info.get("num_col_idx", [])]
    categorical = [columns[int(index)] for index in info.get("cat_col_idx", [])]
    target_indices = [int(index) for index in info.get("target_col_idx", [])]
    target = columns[target_indices[0]] if target_indices else None
    if target is not None:
        (numerical if info.get("task_type") == "regression" else categorical).append(target)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train)); count = len(train)
    discovery = train.iloc[order[:int(.60 * count)]].copy()
    counterexample = train.iloc[order[int(.60 * count):int(.80 * count)]].copy()
    calibration = train.iloc[order[int(.80 * count):]].copy()
    _, mined = empirical_hints(
        audit, args.dataset, discovery, counterexample, categorical, numerical,
        target, audit.is_sensitive, maximum=args.maximum,
    )

    bank, rejected = [], []
    for index, item in enumerate(mined):
        candidate = {key: item[key] for key in CANDIDATE_REQUIRED}
        candidate_id = f"emp_{index:03d}"
        atoms = list(item.get("applicability", [])) + list(item.get("requirements", []))
        touches_target = bool(target) and any(
            atom.get("column", atom.get("left_column")) == target
            or atom.get("right_column") == target
            for atom in atoms
        )
        if args.exclude_target and touches_target:
            rejected.append({**candidate, "candidate_id": candidate_id,
                             "source": "empirical_mining", "reason": "target_column_excluded"})
            continue
        counter_stats = verify(counterexample, item)
        calibration_stats = verify(calibration, item)
        evidence_class = classify_evidence(
            [item["discovery"], counter_stats, calibration_stats],
            args.min_support, conditional=bool(item["applicability"]),
        )
        record = {
            **candidate, "candidate_id": candidate_id, "source": "empirical_mining",
            "canonical": canonical(item), "discovery": item["discovery"],
            "counterexample": counter_stats, "calibration": calibration_stats,
        }
        if evidence_class is None:
            rejected.append({**record, "reason": "three_way_train_validation_failure"})
            continue
        bank.append({**record, "classification": evidence_class,
                     "provenance": ["train_only_empirical_mining"]})

    (args.output / "empirical_bank.json").write_text(json.dumps(bank, indent=2))
    (args.output / "rejected.json").write_text(json.dumps(rejected, indent=2))
    metadata = {
        "method": "train-only empirical implication mining", "llm_used": False,
        "api_calls": 0, "test_file_opened": False, "seed": args.seed,
        "train_rows": len(train), "split_rows": {
            "discovery": len(discovery), "counterexample": len(counterexample),
            "calibration": len(calibration),
        },
        "mined": len(mined), "retained": len(bank), "rejected": len(rejected),
        "exclude_target": args.exclude_target, "target": target,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
