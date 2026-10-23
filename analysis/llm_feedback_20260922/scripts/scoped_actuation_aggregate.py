#!/usr/bin/env python3
"""Aggregate target-scoped actuation runs across seeds and guidance scales."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(args.input_root.glob("scoped_actuation_g*_seed*/scoped_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"no scoped metrics below {args.input_root}")
    data = pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in paths], ignore_index=True)
    data.to_csv(args.output / "all_metrics.csv", index=False)
    group = data.groupby(["guidance_scale", "schedule", "actuation_scope", "split"])
    summary = group.agg(
        seeds=("seed", "nunique"), violation=("conditional_violation_rate", "mean"),
        violation_std=("conditional_violation_rate", "std"), reduction=("conditional_violation_reduction", "mean"),
        antecedent_error=("antecedent_rate_error", "mean"), exception_error=("exception_rate_error", "mean"),
        inactive_error=("inactive_target_mean_error", "mean"), numeric_ks=("global_numeric_ks", "mean"),
        categorical_tv=("global_categorical_tv", "mean"),
    ).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    lines = ["# Target-scoped dynamic actuation", "", "Paired initial noise; same adapter and contracts across scopes.", ""]
    lines.append("| guidance | schedule | scope | split | seeds | violation | reduction | antecedent error | exception error | inactive error | numeric KS | categorical TV |")
    lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.guidance_scale:g} | {row.schedule} | {row.actuation_scope} | {row.split} | {row.seeds} | {row.violation:.4f} | {row.reduction:.4f} | {row.antecedent_error:.4f} | {row.exception_error:.4f} | {row.inactive_error:.4f} | {row.numeric_ks:.4f} | {row.categorical_tv:.4f} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"loaded {len(paths)} runs -> {args.output}")


if __name__ == "__main__":
    main()
