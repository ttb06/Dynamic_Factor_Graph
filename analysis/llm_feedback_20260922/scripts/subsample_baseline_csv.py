#!/usr/bin/env python3
"""Create an equal-size deterministic sample without modifying upstream output."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if len(frame) < args.rows:
        raise ValueError(f"{args.input} has {len(frame)} rows, fewer than requested {args.rows}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.sample(n=args.rows, random_state=args.seed, replace=False).to_csv(args.output, index=False)
    print(f"saved {args.rows} / {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
