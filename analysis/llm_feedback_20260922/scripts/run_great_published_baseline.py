#!/usr/bin/env python3
"""Train and sample the official GReaT implementation on a repository split."""
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
    parser.add_argument("--great-repo", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.great_repo.resolve()))
    from be_great import GReaT

    frame = pd.read_csv(args.train)
    started = time.time()
    model = GReaT(
        args.model, epochs=args.epochs, batch_size=args.batch_size,
        experiment_dir=str(args.output / "trainer"), fp16=True,
        save_strategy="no", logging_steps=100,
    )
    model.fit(frame)
    train_seconds = time.time() - started
    sample_start = time.time()
    synthetic = model.sample(
        n_samples=args.samples, temperature=0.7, k=100,
        max_length=300, drop_nan=False, device="cuda",
        guided_sampling=False,
    )
    sample_seconds = time.time() - sample_start
    synthetic = synthetic.reindex(columns=frame.columns)
    synthetic.to_csv(args.output / "great.csv", index=False)
    summary = {"train_seconds": train_seconds, "sample_seconds": sample_seconds,
               "rows": len(synthetic), "model": args.model, "epochs": args.epochs}
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
