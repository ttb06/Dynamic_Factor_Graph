#!/usr/bin/env python3
"""Expand TabDiff's compact News representation to ConstrainedDGM's layout."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def expand(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    source = {str(column).strip(): column for column in frame.columns}
    result = pd.DataFrame(index=frame.index)
    channel = frame[source["data_channel"]].astype(str).str.lower()
    weekday = frame[source["weekday"]].astype(str).str.lower()
    for name in columns:
        if name in source:
            result[name] = pd.to_numeric(frame[source[name]], errors="raise")
        elif name.startswith("data_channel_is_"):
            result[name] = channel.eq(name.replace("data_channel_is_", "")).astype(np.int8)
        elif name.startswith("weekday_is_"):
            result[name] = weekday.eq(name.replace("weekday_is_", "")).astype(np.int8)
        elif name == "is_weekend":
            result[name] = weekday.isin(["saturday", "sunday"]).astype(np.int8)
        else:
            raise ValueError(f"Cannot reconstruct official News column {name}")
    return result[columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabdiff-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.tabdiff_repo / "data/news/OnlineNewsPopularity/OnlineNewsPopularity.csv", nrows=0)
    columns = [str(column).strip() for column in raw.columns if str(column).strip() != "url"]
    train = expand(pd.read_csv(args.tabdiff_repo / "synthetic/news/real.csv"), columns)
    test = expand(pd.read_csv(args.tabdiff_repo / "synthetic/news/test.csv"), columns)
    # Validation is sampling/evaluation-only in the official driver. Keep the
    # complete TabDiff training split for model fitting and take a deterministic
    # view for the required validation file.
    val = train.sample(n=min(len(test), len(train)), random_state=20261012)
    train.to_csv(args.output / "train_data.csv", index=False)
    val.to_csv(args.output / "val_data.csv", index=False)
    test.to_csv(args.output / "test_data.csv", index=False)
    print({"train": train.shape, "val": val.shape, "test": test.shape})


if __name__ == "__main__":
    main()
