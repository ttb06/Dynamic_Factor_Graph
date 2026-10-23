#!/usr/bin/env python3
"""Apply the public ICLR-2024 constraint layer to a TabDiff News sample."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdgm-repo", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.cdgm_repo))
    from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
    from constraints_code.correct_predictions import correct_preds
    from constraints_code.feature_orderings import set_ordering
    from constraints_code.parser import parse_constraints_file

    frame = pd.read_csv(args.sample)
    # Rebuild the paper's 60-column (URL removed) layout. TabDiff compacts the
    # channel and weekday one-hot fields into two categorical columns, so raw
    # integer positions cease to agree after y_11. Mapping by column name is
    # essential for the y_27/y_28/y_29 self-reference constraints.
    raw_path = args.cdgm_repo.parent / "tabdiff/data/news/OnlineNewsPopularity/OnlineNewsPopularity.csv"
    raw_columns = [str(column).strip() for column in pd.read_csv(raw_path, nrows=0).columns if str(column).strip() != "url"]
    source = {str(column).strip(): column for column in frame.columns}
    matrix = np.zeros((len(frame), len(raw_columns)), dtype=np.float32)
    for index, name in enumerate(raw_columns):
        if name in source:
            matrix[:, index] = pd.to_numeric(frame[source[name]], errors="raise").to_numpy(np.float32)
    if "data_channel" in source:
        values = frame[source["data_channel"]].astype(str).str.lower()
        for index, name in enumerate(raw_columns):
            if name.startswith("data_channel_is_"):
                matrix[:, index] = values.eq(name.replace("data_channel_is_", "")).to_numpy(np.float32)
    if "weekday" in source:
        values = frame[source["weekday"]].astype(str).str.lower()
        for index, name in enumerate(raw_columns):
            if name.startswith("weekday_is_"):
                matrix[:, index] = values.eq(name.replace("weekday_is_", "")).to_numpy(np.float32)
        matrix[:, raw_columns.index("is_weekend")] = values.isin(["saturday", "sunday"]).to_numpy(np.float32)
    ordering, constraints = parse_constraints_file(str(args.cdgm_repo / "data/news/news_constraints.txt"))
    ordering = set_ordering("news", ordering, "corr", "tvae")
    sets = compute_sets_of_constraints(ordering, constraints, verbose=False)
    corrected = correct_preds(torch.as_tensor(matrix, dtype=torch.float32), ordering, sets)
    result = frame.copy()
    corrected_array = corrected.cpu().numpy()
    constrained_names = {
        "n_tokens_title", "n_tokens_content", "n_non_stop_words",
        "n_non_stop_unique_tokens", "num_hrefs", "num_self_hrefs",
        "self_reference_min_shares", "self_reference_max_shares",
        "self_reference_avg_sharess",
    }
    for index, name in enumerate(raw_columns):
        if name in source and name in constrained_names:
            result[source[name]] = corrected_array[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    changed = np.zeros(len(frame), dtype=bool)
    for name in constrained_names:
        if name in source:
            before = pd.to_numeric(frame[source[name]]).to_numpy(float)
            after = pd.to_numeric(result[source[name]]).to_numpy(float)
            changed |= ~np.isclose(before, after, rtol=1e-6, atol=1e-8)
    print({"rows": len(result), "changed_row_fraction": float(changed.mean()), "output": str(args.output)})


if __name__ == "__main__":
    main()
