#!/usr/bin/env python3
"""Train an off-the-shelf SDV baseline on a repository dataset split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", default="shoppers")
    parser.add_argument("--method", choices=["ctgan", "tvae"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    real = pd.read_csv(args.repo / "data" / args.dataset / "train.csv")
    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    metadata = SingleTableMetadata(); metadata.detect_from_dataframe(real)
    categorical = [info["column_names"][i] for i in info["cat_col_idx"] + info["target_col_idx"]]
    for column in categorical: metadata.update_column(column_name=column, sdtype="categorical")
    try: metadata.remove_primary_key()
    except Exception: pass
    common = dict(metadata=metadata, epochs=args.epochs, batch_size=500, verbose=True,
                  enforce_min_max_values=True, enforce_rounding=True, cuda=True)
    model = CTGANSynthesizer(**common) if args.method == "ctgan" else TVAESynthesizer(**common)
    started = time.perf_counter(); model.fit(real); train_seconds = time.perf_counter() - started
    started = time.perf_counter(); synthetic = model.sample(num_rows=args.samples)
    sample_seconds = time.perf_counter() - started
    synthetic = synthetic[real.columns]
    synthetic.to_csv(args.output / "synthetic.csv", index=False)
    model.save(args.output / "model.pkl")
    metadata_out = {"method": args.method.upper(), "dataset": args.dataset,
                    "implementation": "SDV 1.38.3 off-the-shelf synthesizer",
                    "epochs": args.epochs, "samples": args.samples, "seed": args.seed,
                    "train_rows": len(real), "train_seconds": train_seconds,
                    "sample_seconds": sample_seconds, "cuda_available": torch.cuda.is_available()}
    (args.output / "run_metadata.json").write_text(json.dumps(metadata_out, indent=2))
    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__": main()
