#!/usr/bin/env python3
"""Fail-closed summary for the paired six-seed Default SOTA confirmation."""
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

import numpy as np


VARIANTS = ("joint_high", "pair300")
SEEDS = tuple(str(value) for value in range(20261371, 20261377))
SAMPLES = "8192"
STEPS = "50"
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
        result = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: nonnumeric {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context}: nonfinite {value!r}")
    return result


def read_env(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            output[key.strip()] = value.strip()
    return output


def read_hash(path: Path) -> str:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    values = [line.split()[0].casefold() for line in lines]
    if len(values) != 2 or len(set(values)) != 1 or len(values[0]) != 64:
        raise ValueError(f"{path}: invalid paired hashes")
    return values[0]


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
    columns = {key: resolve(fieldnames, values) for key, values in aliases.items()}
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        method = str(row[method_column]).strip()
        if not method or method in output:
            raise ValueError(f"{path}: blank/duplicate method")
        output[method] = {
            key: finite(row[column], f"{path}:{method}:{column}")
            for key, column in columns.items()
        }
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


def bootstrap_interval(values: Sequence[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(20000, len(array)))].mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def load(pattern: str) -> dict[tuple[str, str], dict[str, Any]]:
    paths = sorted(Path(value).resolve() for value in glob.glob(pattern) if Path(value).is_dir())
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            variant = provenance.get("variant", "")
            seed = provenance.get("seed", "")
            key = (variant, seed)
            if variant not in VARIANTS or seed not in SEEDS or key in runs:
                raise ValueError(f"invalid/duplicate design cell {key!r}")
            if provenance.get("dataset") != "default":
                raise ValueError("dataset mismatch")
            if provenance.get("samples") != SAMPLES or provenance.get("steps") != STEPS:
                raise ValueError("sample/step mismatch")
            if provenance.get("exit_code") != "0":
                raise ValueError("missing successful exit marker")
            metrics = read_metrics(path / "eval" / "baseline_metrics.csv")
            if set(metrics) != {"B", "G", variant}:
                raise ValueError(f"unexpected methods {sorted(metrics)}")
            candidate_metadata = json.loads((path / "candidate" / "run_metadata.json").read_text())
            reference_metadata = json.loads((path / "reference" / "run_metadata.json").read_text())
            runtime_ratio = finite(candidate_metadata.get("runtime_seconds"), "candidate runtime") / finite(reference_metadata.get("runtime_seconds"), "reference runtime")
            runs[key] = {
                "path": str(path), "provenance": provenance,
                "metrics": metrics, "paired_hash": read_hash(path / "unconditional_sha256.txt"),
                "runtime_ratio": runtime_ratio,
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if len(paths) != len(expected) or set(runs) != expected:
        errors.append(f"matched={len(paths)}; missing={sorted(expected-set(runs))}; extra={sorted(set(runs)-expected)}")
    if errors:
        raise ValueError("invalid confirmation round:\n- " + "\n- ".join(errors))
    return runs


def assess(variant: str, runs: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        run = runs[(variant, seed)]
        metrics = run["metrics"]
        per_seed.append({
            "seed": seed,
            "versus_g": benefit(metrics[variant], metrics["G"]),
            "versus_b": benefit(metrics[variant], metrics["B"]),
            "candidate": metrics[variant], "g": metrics["G"],
            "runtime_ratio": run["runtime_ratio"],
            "paired_hash": run["paired_hash"], "path": run["path"],
        })
    keys = ("shape", "trend", "alpha", "beta", "auc", "detection", "dcr")
    mean_g = {key: float(np.mean([row["versus_g"][key] for row in per_seed])) for key in keys}
    mean_b = {key: float(np.mean([row["versus_b"][key] for row in per_seed])) for key in keys}
    wins_g = {key: int(sum(row["versus_g"][key] > 0 for row in per_seed)) for key in keys}
    ci_g = {key: bootstrap_interval([row["versus_g"][key] for row in per_seed], 20261390 + index) for index, key in enumerate(keys)}
    raw_candidate = {key: float(np.mean([row["candidate"][key] for row in per_seed])) for key in keys}
    reasons: list[str] = []
    for key in ("shape", "trend"):
        if mean_g[key] <= 0:
            reasons.append(f"mean {key} benefit vs G {mean_g[key]:.6f} is not positive")
        if wins_g[key] < 4:
            reasons.append(f"{key} wins vs G {wins_g[key]}/6 < 4/6")
        if mean_b[key] <= 0:
            reasons.append(f"mean {key} benefit vs B {mean_b[key]:.6f} is not positive")
    floors = {"alpha": -0.05, "beta": -0.15, "auc": -0.005, "dcr": -0.50}
    for key, floor in floors.items():
        if mean_g[key] < floor:
            reasons.append(f"mean {key} benefit vs G {mean_g[key]:.6f} < {floor}")
    runtime = float(np.mean([row["runtime_ratio"] for row in per_seed]))
    if runtime > 1.5:
        reasons.append(f"mean runtime ratio {runtime:.3f} > 1.5")
    score = 0.40 * mean_g["shape"] + 0.35 * mean_g["trend"] + 0.15 * mean_g["alpha"] + 5.0 * mean_g["auc"] + 0.05 * mean_g["beta"]
    return {
        "variant": variant, "confirmed": not reasons,
        "rejection_reasons": reasons, "score": score,
        "mean_versus_g_oriented": mean_g, "mean_versus_b_oriented": mean_b,
        "wins_versus_g": wins_g, "bootstrap_95pct_mean_versus_g": ci_g,
        "mean_candidate_raw": raw_candidate, "mean_runtime_ratio": runtime,
        "candidate_beats_local_tabsyn_shape_on_mean": raw_candidate["shape"] < TAB_SYN["shape"],
        "candidate_beats_local_tabsyn_trend_on_mean": raw_candidate["trend"] < TAB_SYN["trend"],
        "per_seed": per_seed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = load(args.input_glob)
    variants = [assess(variant, runs) for variant in VARIANTS]
    variants.sort(key=lambda item: (not item["confirmed"], -item["score"], item["variant"]))
    confirmed = [item["variant"] for item in variants if item["confirmed"]]
    winner = confirmed[0] if confirmed else None
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True,
        "seeds": list(SEEDS),
        "published_context_only": {"TabSyn_local_one_run": TAB_SYN},
        "confirmed_variants": confirmed, "recommended_variant": winner,
        "variants": variants,
    }
    (args.output / "confirmation.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Default hybrid SOTA six-seed confirmation", "",
        f"- Confirmed: **{', '.join(confirmed) if confirmed else 'none'}**",
        f"- Recommended: **{winner or 'none'}**", "",
        "All deltas are benefit-oriented candidate minus its same-node paired G; lower Shape/Trend is converted to positive benefit.", "",
        "| Rank | Arm | Confirmed | Score | dShape | wins | 95% CI | dTrend | wins | 95% CI | dAlpha | dBeta | dAUC | Runtime x | Mean Shape | Mean Trend |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, item in enumerate(variants, start=1):
        delta = item["mean_versus_g_oriented"]
        wins = item["wins_versus_g"]
        ci = item["bootstrap_95pct_mean_versus_g"]
        raw = item["mean_candidate_raw"]
        lines.append(
            f"| {rank} | {item['variant']} | {'yes' if item['confirmed'] else 'no'} | {item['score']:.6f} | {delta['shape']:.6f} | {wins['shape']}/6 | [{ci['shape'][0]:.6f}, {ci['shape'][1]:.6f}] | {delta['trend']:.6f} | {wins['trend']}/6 | [{ci['trend'][0]:.6f}, {ci['trend'][1]:.6f}] | {delta['alpha']:.6f} | {delta['beta']:.6f} | {delta['auc']:.6f} | {item['mean_runtime_ratio']:.3f} | {raw['shape']:.6f} | {raw['trend']:.6f} |"
        )
    lines.extend(["", "## Seed-level primary deltas", "", "| Arm | Seed | dShape | dTrend | dAlpha | dBeta | dAUC |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for item in variants:
        for row in item["per_seed"]:
            delta = row["versus_g"]
            lines.append(f"| {item['variant']} | {row['seed']} | {delta['shape']:.6f} | {delta['trend']:.6f} | {delta['alpha']:.6f} | {delta['beta']:.6f} | {delta['auc']:.6f} |")
    lines.extend(["", "## Rejections", ""])
    for item in variants:
        if not item["confirmed"]:
            lines.append(f"- **{item['variant']}**: " + "; ".join(item["rejection_reasons"]))
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.output / "SUMMARY.md")
    print(f"Confirmed: {', '.join(confirmed) or 'none'}; recommended: {winner or 'none'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
