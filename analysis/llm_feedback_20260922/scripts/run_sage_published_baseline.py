#!/usr/bin/env python3
"""Run the released SAGE implementation and save both paper guidance variants."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sage-repo", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, str(args.sage_repo.resolve()))
    from sage import SAGE

    config = vars(args).copy()
    config.update({k: str(v) for k, v in config.items() if isinstance(v, Path)})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = SAGE(model_name=args.model, device=torch.device("cuda:0"), use_lora=args.use_lora)
    started = time.time()
    checkpoint_path = args.checkpoint_path or (args.output / "best_model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        str(args.train), epochs=args.epochs, batch_size=args.batch_size, lr=1e-4,
        max_length=300, val_ratio=0.001, early_stopping_rounds=5,
        mi_n_bins=5, mi_strategy="fd", mi_threshold=None,
        constrain_string_values=True, num_workers=4,
        gradient_accumulation_steps=1, shuffle=True, seed=args.seed,
        deterministic=False, checkpoint_path=str(checkpoint_path),
    )
    train_seconds = time.time() - started

    fs_start = time.time()
    fs = model.sample(
        sample_num=args.samples, p=0.7, temperature=1.0,
        mi_threshold=None, copy_factor=1, apply_final_constraints=True,
    )
    fs_seconds = time.time() - fs_start
    fs.to_csv(args.output / "sage_feature_selector.csv", index=False)

    lc_start = time.time()
    lc = model.sample_logits(
        sample_num=args.samples, p=0.7, temperature=1.0, mi_lambda=1.0,
        scale_clip_min=0.5, scale_clip_max=1.5,
        apply_final_constraints=True,
    )
    lc_seconds = time.time() - lc_start
    lc.to_csv(args.output / "sage_logit_correction.csv", index=False)
    summary = {
        "train_seconds": train_seconds,
        "feature_selector_seconds": fs_seconds,
        "logit_correction_seconds": lc_seconds,
        "feature_selector_rows": len(fs),
        "logit_correction_rows": len(lc),
        "paper_protocol_notes": {
            "prefix_fraction_p": 0.7,
            "temperature": 1.0,
            "mi_bins_adult": 5,
            "mi_threshold": "training median",
            "important": "SAGE initializes 70% of each record from one real training row, as released code/paper protocol does.",
        },
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
