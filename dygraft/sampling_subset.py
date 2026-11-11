#!/usr/bin/env python3
"""Select up to a requested row count without duplicating short outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    selected = min(args.rows, len(frame))
    if len(frame) > selected:
        frame = frame.sample(n=selected, random_state=args.seed, replace=False)
    frame = frame.reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.report.write_text(json.dumps({
        "source": str(args.input), "source_rows": len(pd.read_csv(args.input, usecols=[0])),
        "requested_rows": args.rows, "selected_rows": selected,
        "sampling_with_replacement": False, "seed": args.seed,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
