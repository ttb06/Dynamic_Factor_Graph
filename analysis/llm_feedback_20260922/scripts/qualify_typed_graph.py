#!/usr/bin/env python3
"""Fail closed unless a curated graph is stable and readable enough for denoising actuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curation", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    curation = json.loads((args.curation/"summary.json").read_text())
    schedule = pd.read_csv(args.trajectory/"typed_readability_schedule.csv")
    errors = (schedule.synthetic_violation_rate-schedule.heldout_violation_rate).abs().replace([np.inf,-np.inf],np.nan)
    checks = {
        "at_least_three_active_edges": curation["active_edges"] >= 3,
        "at_least_two_hard_edges": curation["hard"] >= 2,
        "has_column_comparison": curation["column_comparisons"] >= 1,
        "at_least_two_readable_edges": int(schedule.readable.sum()) >= 2,
        "at_least_two_readable_by_middle": int((schedule.readable & schedule.cutoff.ge(.25)).sum()) >= 2,
        "mean_synthetic_heldout_error_le_005": bool(errors.mean() <= .05),
    }
    passed = all(checks.values())
    result = {"qualified": passed, "checks": checks, "curation": curation,
              "readable_edges": int(schedule.readable.sum()),
              "readable_by_middle": int((schedule.readable & schedule.cutoff.ge(.25)).sum()),
              "mean_synthetic_heldout_error": float(errors.mean())}
    (args.output/"qualification.json").write_text(json.dumps(result, indent=2))
    lines = ["# Typed graph qualification", "", f"**Qualified for denoising actuation: {passed}**", "",
             "| Check | Passed |", "| --- | --- |"]
    lines += [f"| {key} | {value} |" for key,value in checks.items()]
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result, indent=2))
    if not passed: raise SystemExit(3)


if __name__ == "__main__": main()
