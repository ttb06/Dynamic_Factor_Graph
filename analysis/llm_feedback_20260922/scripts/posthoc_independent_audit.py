#!/usr/bin/env python3
"""Evaluate constraints and exception subgroups not used by the sampler proxies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype(str).str.strip()
    return result


def metrics(name: str, raw: pd.DataFrame) -> dict:
    df = clean(raw)
    work_missing = df["workclass"].eq("?")
    occupation_missing = df["occupation"].eq("?")
    private_house = df["occupation"].eq("Priv-house-serv")
    own_child = df["relationship"].eq("Own-child")
    preschool = df["education"].eq("Preschool")
    age = pd.to_numeric(df["age"], errors="coerce")
    return {
        "variant": name,
        "rows": len(df),
        # These relations were discovered/audited on train but never used by
        # any guidance, repair, or best-of-N score in tabdiff_probe.py.
        "withheld_workclass_missing_to_occupation_missing_violation": float(
            (work_missing & ~occupation_missing).mean()
        ),
        "withheld_missing_pair_xor_rate": float((work_missing ^ occupation_missing).mean()),
        "withheld_private_house_to_private_workclass_violation": float(
            (private_house & ~df["workclass"].eq("Private")).mean()
        ),
        "private_house_applicable_rows": int(private_house.sum()),
        # Exception-bearing soft relations are distribution diagnostics, not
        # hard validity metrics.
        "own_child_rows": int(own_child.sum()),
        "own_child_age_ge_30_conditional_rate": float((age[own_child] >= 30).mean()) if own_child.any() else float("nan"),
        "preschool_rows": int(preschool.sum()),
        "preschool_high_income_conditional_rate": float(df.loc[preschool, "income"].eq(">50K").mean()) if preschool.any() else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [metrics("real_train", pd.read_csv(args.real)), metrics("real_test", pd.read_csv(args.test))]
    for path in sorted(args.samples.glob("samples_*.csv")):
        rows.append(metrics(path.stem.removeprefix("samples_"), pd.read_csv(path)))
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
