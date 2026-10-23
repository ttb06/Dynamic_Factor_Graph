#!/usr/bin/env python3
"""Fail-closed selector for the Default balanced-particle screen."""
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
    "uniform32", "rake_marg35", "rake_pair35", "rake_pair60",
    "ratio_logit50", "ratio_xgb50", "hybrid_logit50", "hybrid_xgb15",
)
SEED = "20261380"
POOL_SAMPLES = 16384
SELECTED_SAMPLES = 8192
STEPS = 50


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


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    aliases = {
        "shape": ("shape_error_percent",), "trend": ("trend_error_percent",),
        "alpha": ("alpha_precision_percent",), "beta": ("beta_recall_percent",),
        "auc": ("mle_xgboost_auc",), "detection": ("detection_score",),
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
            if provenance.get("pool_samples") != str(POOL_SAMPLES) or provenance.get("selected_samples") != str(SELECTED_SAMPLES) or provenance.get("steps") != str(STEPS):
                raise ValueError("design mismatch")
            if provenance.get("exit_code") != "0":
                raise ValueError("missing successful exit marker")
            metrics = read_metrics(path / "eval" / "baseline_metrics.csv")
            if set(metrics) != {"B", "G", "R", variant}:
                raise ValueError(f"unexpected methods {sorted(metrics)}")
            metadata = json.loads((path / "resampled" / "selection_metadata.json").read_text())
            if metadata.get("dataset") != "default" or int(metadata.get("seed")) != int(SEED):
                raise ValueError("resampler metadata mismatch")
            if metadata.get("pool_rows") != POOL_SAMPLES or metadata.get("selected_rows") != SELECTED_SAMPLES:
                raise ValueError("resampler row-count mismatch")
            if metadata.get("no_duplicate_selected_rows") is not True:
                raise ValueError("selected rows contain duplicate pool indices")
            best = metadata["selection"]["best"]
            uniform = metadata["selection"]["uniform_control"]
            if finite(best["objective"], "best objective") > finite(uniform["objective"], "uniform objective") + 1e-12:
                raise ValueError("best draw is worse than internal uniform control")
            runs[variant] = {
                "path": str(path), "provenance": provenance,
                "metrics": metrics, "metadata": metadata,
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if len(paths) != len(VARIANTS) or set(runs) != set(VARIANTS):
        errors.append(f"matched={len(paths)}; missing={sorted(set(VARIANTS)-set(runs))}")
    if errors:
        raise ValueError("invalid balanced-particle round:\n- " + "\n- ".join(errors))
    return runs


def assess(variant: str, run: Mapping[str, Any]) -> dict[str, Any]:
    metrics = run["metrics"]
    versus_g = benefit(metrics[variant], metrics["G"])
    versus_r = benefit(metrics[variant], metrics["R"])
    versus_b = benefit(metrics[variant], metrics["B"])
    metadata = run["metadata"]
    best_objective = finite(metadata["selection"]["best"]["objective"], "best objective")
    uniform_objective = finite(metadata["selection"]["uniform_control"]["objective"], "uniform objective")
    objective_gain = uniform_objective - best_objective
    reasons: list[str] = []
    for reference, values in (("G", versus_g), ("R", versus_r)):
        for metric in ("shape", "trend"):
            if values[metric] <= 0:
                reasons.append(f"{metric} benefit vs {reference} {values[metric]:.6f} is not positive")
    for metric in ("shape", "trend"):
        if versus_b[metric] <= 0:
            reasons.append(f"{metric} benefit vs B {versus_b[metric]:.6f} is not positive")
    floors = {"alpha": -0.08, "beta": -0.20, "auc": -0.005, "dcr": -0.75}
    for metric, floor in floors.items():
        if versus_g[metric] < floor:
            reasons.append(f"{metric} benefit vs G {versus_g[metric]:.6f} < {floor}")
    if objective_gain <= 0:
        reasons.append(f"moment objective gain {objective_gain:.6f} is not positive")
    pre_eval_runtime = finite(run["provenance"].get("pre_eval_runtime_seconds"), "pre-eval runtime")
    if pre_eval_runtime > 3600:
        reasons.append(f"pre-eval runtime {pre_eval_runtime:.1f}s > 3600s")
    score = 0.40 * versus_g["shape"] + 0.35 * versus_g["trend"] + 0.15 * versus_g["alpha"] + 5.0 * versus_g["auc"] + 0.05 * versus_g["beta"]
    return {
        "variant": variant, "eligible": not reasons,
        "rejection_reasons": reasons, "score": score,
        "versus_g_oriented": versus_g, "versus_r_oriented": versus_r,
        "versus_b_oriented": versus_b,
        "candidate_raw": metrics[variant], "g_raw": metrics["G"], "r_raw": metrics["R"],
        "moment_objective_gain_over_uniform": objective_gain,
        "best_moment_objective": best_objective,
        "uniform_moment_objective": uniform_objective,
        "effective_sample_size": finite(metadata["selection"]["effective_sample_size"], "ESS"),
        "pre_eval_runtime_seconds": pre_eval_runtime,
        "mode": metadata.get("mode"), "parameters": metadata.get("parameters"),
        "path": run["path"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-winners", type=int, default=2)
    args = parser.parse_args(argv)
    runs = load(args.input_glob)
    candidates = [assess(variant, runs[variant]) for variant in VARIANTS]
    candidates.sort(key=lambda item: (not item["eligible"], -item["score"], item["variant"]))
    winners: list[str] = []
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
        item["selected"] = item["eligible"] and len(winners) < args.max_winners
        if item["selected"]:
            winners.append(item["variant"])
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True, "seed": SEED, "winners": winners,
        "candidates": candidates,
    }
    (args.output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Default balanced-particle selection", "",
        f"- Selected: **{', '.join(winners) if winners else 'none'}**", "",
        "| Rank | Selected | Arm | Eligible | Score | dShape/G | dTrend/G | dShape/R | dTrend/R | dAlpha/G | dBeta/G | dAUC/G | Moment gain | ESS | Runtime s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in candidates:
        dg = item["versus_g_oriented"]
        dr = item["versus_r_oriented"]
        lines.append(
            f"| {item['rank']} | {'yes' if item['selected'] else ''} | {item['variant']} | {'yes' if item['eligible'] else 'no'} | {item['score']:.6f} | {dg['shape']:.6f} | {dg['trend']:.6f} | {dr['shape']:.6f} | {dr['trend']:.6f} | {dg['alpha']:.6f} | {dg['beta']:.6f} | {dg['auc']:.6f} | {item['moment_objective_gain_over_uniform']:.6f} | {item['effective_sample_size']:.1f} | {item['pre_eval_runtime_seconds']:.0f} |"
        )
    lines.extend(["", "## Rejections", ""])
    for item in candidates:
        if not item["eligible"]:
            lines.append(f"- **{item['variant']}**: " + "; ".join(item["rejection_reasons"]))
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
