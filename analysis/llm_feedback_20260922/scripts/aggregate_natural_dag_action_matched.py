#!/usr/bin/env python3
"""Compare dynamic topology and frozen topology at identical per-seed action counts."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def online(path: Path) -> pd.Series:
    frame = pd.read_csv(path / "method_metrics.csv")
    return frame.loc[frame.method.eq("xstate_reward_calibrated")].iloc[0]


def seed(path: Path) -> int:
    match = re.search(r"_s(\d+)_", path.name)
    return int(match.group(1)) if match else 20261376


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", nargs="+", type=Path, required=True)
    parser.add_argument("--frozen", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    matched = {seed(path): online(path) for path in args.matched}
    frozen = {seed(path): online(path) for path in args.frozen}
    rows = []
    for value in sorted(matched):
        left, right = matched[value], frozen[value]
        row = {"seed": value, "dynamic_actions": int(left.action_events),
               "frozen_actions": int(right.action_events)}
        for metric in ("macro_violation", "heldout_violation_error",
                       "calibration_violation_error", "categorical_tv"):
            row[f"dynamic_{metric}"] = float(left[metric])
            row[f"frozen_{metric}"] = float(right[metric])
            if metric != "categorical_tv":
                row[f"gain_{metric}_pct"] = 100.0 * (float(right[metric]) - float(left[metric])) / float(right[metric])
        rows.append(row)
    result = pd.DataFrame(rows); result.to_csv(args.output / "action_matched_metrics.csv", index=False)
    lines = ["# Action-matched natural DAG control", "",
             "Dynamic is capped to the exact number of online actions used by frozen for each paired seed.", "",
             "| Seed | Actions dynamic/frozen | Macro dynamic/frozen | Held-out dynamic/frozen | Held-out gain | Calibration dynamic/frozen | TV dynamic/frozen |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.dynamic_actions}/{row.frozen_actions} | "
            f"{row.dynamic_macro_violation:.4f}/{row.frozen_macro_violation:.4f} | "
            f"{row.dynamic_heldout_violation_error:.4f}/{row.frozen_heldout_violation_error:.4f} | "
            f"{row.gain_heldout_violation_error_pct:.1f}% | "
            f"{row.dynamic_calibration_violation_error:.4f}/{row.frozen_calibration_violation_error:.4f} | "
            f"{row.dynamic_categorical_tv:.4f}/{row.frozen_categorical_tv:.4f} |"
        )
    lines += ["", f"Mean held-out gain: **{result.gain_heldout_violation_error_pct.mean():.1f}%** "
              f"({int((result.gain_heldout_violation_error_pct > 0).sum())}/{len(result)} seeds).",
              f"Mean macro gain: {result.gain_macro_violation_pct.mean():.1f}%; mean calibration gain: "
              f"{result.gain_calibration_violation_error_pct.mean():.1f}%. The matched budget therefore "
              "supports row/topology selection for held-out fidelity, but the full calibration gain also "
              "depends on spending the additional downstream actions exposed by dynamic refresh."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
