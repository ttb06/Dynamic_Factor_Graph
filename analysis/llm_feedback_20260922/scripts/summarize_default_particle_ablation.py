#!/usr/bin/env python3
"""Aggregate six-seed balanced-particle mechanism ablations."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


SEEDS = tuple(str(value) for value in range(20261381, 20261387))
METHODS = ("B", "G", "R", "uniform32", "rake_marg35", "ratio_xgb50")
METRICS = ("shape", "trend", "alpha", "beta", "auc", "detection", "dcr")


def norm(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def resolve(names: list[str], alias: str) -> str:
    indexed = {norm(name): name for name in names}
    if norm(alias) not in indexed:
        raise ValueError(f"missing {alias!r} from {names!r}")
    return indexed[norm(alias)]


def finite(value: object) -> float:
    result = float(str(value).strip().rstrip("%"))
    if not math.isfinite(result):
        raise ValueError(f"nonfinite value {value!r}")
    return result


def read_env(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            output[key.strip()] = value.strip()
    return output


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    aliases = {
        "shape": "shape_error_percent", "trend": "trend_error_percent",
        "alpha": "alpha_precision_percent", "beta": "beta_recall_percent",
        "auc": "mle_xgboost_auc", "detection": "detection_score",
        "dcr": "dcr_percent_50_ideal",
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        rows = list(reader)
    method_column = resolve(names, "method")
    columns = {key: resolve(names, alias) for key, alias in aliases.items()}
    output = {
        str(row[method_column]).strip(): {key: finite(row[column]) for key, column in columns.items()}
        for row in rows
    }
    if set(output) != set(METHODS):
        raise ValueError(f"{path}: methods={sorted(output)}")
    return output


def oriented(left: dict[str, float], right: dict[str, float], metric: str) -> float:
    if metric in {"shape", "trend"}:
        return right[metric] - left[metric]
    if metric == "dcr":
        return abs(right[metric] - 50.0) - abs(left[metric] - 50.0)
    return left[metric] - right[metric]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = sorted(Path(value).resolve() for value in glob.glob(args.input_glob) if Path(value).is_dir())
    runs: dict[str, dict[str, dict[str, float]]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            seed = provenance.get("seed", "")
            if seed not in SEEDS or seed in runs or provenance.get("exit_code") != "0":
                raise ValueError(f"invalid seed/exit {seed!r}")
            for subdir in ("uniform32", "rake_marg35"):
                metadata = json.loads((path / subdir / "selection_metadata.json").read_text())
                if metadata.get("no_duplicate_selected_rows") is not True or metadata.get("selected_rows") != 8192:
                    raise ValueError(f"invalid {subdir} selection")
            runs[seed] = read_metrics(path / "eval" / "baseline_metrics.csv")
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if len(paths) != len(SEEDS) or set(runs) != set(SEEDS):
        errors.append(f"matched={len(paths)}; missing={sorted(set(SEEDS)-set(runs))}")
    if errors:
        raise ValueError("invalid ablation round:\n- " + "\n- ".join(errors))
    rows = []
    for method in METHODS:
        raw = {metric: float(np.mean([runs[seed][method][metric] for seed in SEEDS])) for metric in METRICS}
        delta = {metric: float(np.mean([oriented(runs[seed][method], runs[seed]["G"], metric) for seed in SEEDS])) for metric in METRICS}
        wins = {metric: int(sum(oriented(runs[seed][method], runs[seed]["G"], metric) > 0 for seed in SEEDS)) for metric in METRICS}
        rows.append({"method": method, "mean_raw": raw, "mean_versus_g_oriented": delta, "wins_versus_g": wins})
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS), "methods": rows}
    (args.output / "ablation.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Default balanced-particle multi-seed ablation", "",
        "| Method | Mean Shape | Mean Trend | Mean Alpha | Mean Beta | Mean AUC | dShape/G | wins | dTrend/G | wins | dAUC/G |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        raw, delta, wins = row["mean_raw"], row["mean_versus_g_oriented"], row["wins_versus_g"]
        lines.append(f"| {row['method']} | {raw['shape']:.6f} | {raw['trend']:.6f} | {raw['alpha']:.6f} | {raw['beta']:.6f} | {raw['auc']:.6f} | {delta['shape']:.6f} | {wins['shape']}/6 | {delta['trend']:.6f} | {wins['trend']}/6 | {delta['auc']:.6f} |")
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.output / "SUMMARY.md")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
