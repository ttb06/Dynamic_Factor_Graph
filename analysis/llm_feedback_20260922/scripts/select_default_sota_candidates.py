#!/usr/bin/env python3
"""Fail-closed selector for the paired Default hybrid SOTA local search."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


VARIANTS = (
    "num035", "num065", "cat015", "cat035", "marg005", "marg020",
    "pair150", "pair300", "tail005", "tail015", "joint_low", "joint_high",
)
SEED = "20261370"
SAMPLES = 8192
STEPS = 50
TAB_SYN = {"shape": 1.4810, "trend": 1.6499, "alpha": 99.2198, "beta": 47.7816, "auc": 0.7127}


def norm(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def resolve(fieldnames: list[str], aliases: tuple[str, ...]) -> str:
    indexed = {norm(name): name for name in fieldnames}
    matches = list(dict.fromkeys(indexed[norm(alias)] for alias in aliases if norm(alias) in indexed))
    if len(matches) != 1:
        raise ValueError(f"cannot resolve {aliases!r} from {fieldnames!r}")
    return matches[0]


def finite(value: object, context: str) -> float:
    try:
        output = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: non-numeric {value!r}") from exc
    if not math.isfinite(output):
        raise ValueError(f"{context}: non-finite {value!r}")
    return output


def read_env(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            output[key.strip()] = value.strip()
    return output


def read_hash(path: Path) -> str:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    digests = [line.split()[0].casefold() for line in lines]
    if len(digests) != 2 or len(set(digests)) != 1 or len(digests[0]) != 64:
        raise ValueError(f"{path}: invalid paired hashes")
    return digests[0]


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    aliases = {
        "shape": ("shape_error_percent",),
        "trend": ("trend_error_percent",),
        "alpha": ("alpha_precision_percent",),
        "beta": ("beta_recall_percent",),
        "auc": ("mle_xgboost_auc",),
        "detection": ("detection_score",),
        "dcr": ("dcr_percent_50_ideal",),
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    method_column = resolve(fieldnames, ("method",))
    columns = {key: resolve(fieldnames, names) for key, names in aliases.items()}
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        method = str(row[method_column]).strip()
        if not method or method in output:
            raise ValueError(f"{path}: blank/duplicate method")
        output[method] = {key: finite(row[column], f"{path}:{method}:{column}") for key, column in columns.items()}
    return output


def benefit(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {
        "shape": right["shape"] - left["shape"],
        "trend": right["trend"] - left["trend"],
        "alpha": left["alpha"] - right["alpha"],
        "beta": left["beta"] - right["beta"],
        "auc": left["auc"] - right["auc"],
        "detection": left["detection"] - right["detection"],
        "dcr": abs(right["dcr"] - 50.0) - abs(left["dcr"] - 50.0),
    }


def load(pattern: str) -> dict[str, dict[str, Any]]:
    paths = sorted(Path(value).resolve() for value in glob.glob(pattern) if Path(value).is_dir())
    runs: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            variant = provenance.get("variant", "")
            if variant not in VARIANTS or variant in runs:
                raise ValueError(f"invalid/duplicate variant {variant!r}")
            if provenance.get("dataset") != "default" or provenance.get("seed") != SEED:
                raise ValueError("dataset/seed mismatch")
            if provenance.get("samples") != str(SAMPLES) or provenance.get("steps") != str(STEPS):
                raise ValueError("design mismatch")
            metrics = read_metrics(path / "eval" / "baseline_metrics.csv")
            if set(metrics) != {"B", "G", variant}:
                raise ValueError(f"methods={sorted(metrics)}")
            metadata = json.loads((path / "candidate" / "run_metadata.json").read_text())
            reference_metadata = json.loads((path / "reference" / "run_metadata.json").read_text())
            runs[variant] = {
                "path": str(path),
                "provenance": provenance,
                "metrics": metrics,
                "hash": read_hash(path / "unconditional_sha256.txt"),
                "runtime_ratio": finite(metadata.get("runtime_seconds"), "candidate runtime") / finite(reference_metadata.get("runtime_seconds"), "reference runtime"),
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if len(paths) != len(VARIANTS) or set(runs) != set(VARIANTS):
        errors.append(f"matched={len(paths)}, missing={sorted(set(VARIANTS)-set(runs))}")
    if errors:
        raise ValueError("invalid Default SOTA round:\n- " + "\n- ".join(errors))
    return runs


def assess(variant: str, run: Mapping[str, Any]) -> dict[str, Any]:
    values = run["metrics"]
    versus_g = benefit(values[variant], values["G"])
    versus_b = benefit(values[variant], values["B"])
    reasons: list[str] = []
    if versus_g["shape"] <= 0:
        reasons.append(f"Shape benefit vs G {versus_g['shape']:.6f} is not positive")
    if versus_g["trend"] <= 0:
        reasons.append(f"Trend benefit vs G {versus_g['trend']:.6f} is not positive")
    floors = {"alpha": -0.05, "beta": -0.15, "auc": -0.005, "dcr": -0.50}
    for metric, floor in floors.items():
        if versus_g[metric] < floor:
            reasons.append(f"{metric} benefit vs G {versus_g[metric]:.6f} < {floor}")
    if run["runtime_ratio"] > 1.5:
        reasons.append(f"runtime {run['runtime_ratio']:.3f}x > 1.5x G")
    denominator_shape = values["G"]["shape"] - TAB_SYN["shape"]
    denominator_trend = values["G"]["trend"] - TAB_SYN["trend"]
    shape_closure = versus_g["shape"] / denominator_shape if denominator_shape > 0 else 0.0
    trend_closure = versus_g["trend"] / denominator_trend if denominator_trend > 0 else 0.0
    score = (
        0.40 * versus_g["shape"] + 0.35 * versus_g["trend"]
        + 0.15 * versus_g["alpha"] + 5.0 * versus_g["auc"]
        + 0.05 * versus_g["beta"]
    )
    return {
        "variant": variant,
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "score": score,
        "versus_g_oriented": versus_g,
        "versus_b_oriented": versus_b,
        "candidate_raw": values[variant],
        "g_raw": values["G"],
        "shape_gap_closure_to_local_tabsyn": shape_closure,
        "trend_gap_closure_to_local_tabsyn": trend_closure,
        "candidate_beats_local_tabsyn_shape": values[variant]["shape"] < TAB_SYN["shape"],
        "candidate_beats_local_tabsyn_trend": values[variant]["trend"] < TAB_SYN["trend"],
        "runtime_ratio": run["runtime_ratio"],
        "paired_hash": run["hash"],
        "parameters": {key: run["provenance"].get(key) for key in ("numeric_strength", "categorical_strength", "marginal_scale", "pair_scale", "tail")},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-winners", type=int, default=2)
    args = parser.parse_args(argv)
    runs = load(args.input_glob)
    candidates = [assess(variant, runs[variant]) for variant in VARIANTS]
    candidates.sort(key=lambda value: (not value["eligible"], -value["score"], value["variant"]))
    winners: list[str] = []
    for rank, value in enumerate(candidates, start=1):
        value["rank"] = rank
        value["selected"] = value["eligible"] and len(winners) < args.max_winners
        if value["selected"]:
            winners.append(value["variant"])
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True,
        "seed": SEED,
        "published_context_only": {"TabSyn_local_one_run": TAB_SYN},
        "winners": winners,
        "candidates": candidates,
        "input_directories": [runs[variant]["path"] for variant in VARIANTS],
    }
    (args.output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Default hybrid SOTA local-search selection", "",
        f"- Selected: **{', '.join(winners) if winners else 'none'}**", "",
        "TabSyn numbers are local one-run context only; promotion is paired against G.", "",
        "| Rank | Selected | Arm | Eligible | Score | dShape/G | dTrend/G | dAlpha/G | dBeta/G | dAUC/G | Shape gap closure | Runtime x |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for value in candidates:
        delta = value["versus_g_oriented"]
        lines.append(
            f"| {value['rank']} | {'yes' if value['selected'] else ''} | {value['variant']} | {'yes' if value['eligible'] else 'no'} | {value['score']:.6f} | {delta['shape']:.6f} | {delta['trend']:.6f} | {delta['alpha']:.6f} | {delta['beta']:.6f} | {delta['auc']:.6f} | {value['shape_gap_closure_to_local_tabsyn']:.3f} | {value['runtime_ratio']:.3f} |"
        )
    lines.extend(["", "## Rejections", ""])
    for value in candidates:
        if not value["eligible"]:
            lines.append(f"- **{value['variant']}**: " + "; ".join(value["rejection_reasons"]))
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.output / "SUMMARY.md")
    print(f"Selected {len(winners)}: {', '.join(winners) or 'none'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
