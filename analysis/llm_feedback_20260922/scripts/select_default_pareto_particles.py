#!/usr/bin/env python3
"""Fail-closed multi-seed selector for train-only Pareto particle arms."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


VARIANTS = ("pw075", "pw150", "pw300", "cov150", "tail20", "temp50", "target16", "uniform150")
SEEDS = tuple(str(value) for value in range(20261391, 20261397))
METRICS = ("shape", "trend", "alpha", "beta", "auc", "detection", "dcr")


def norm(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


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
        "shape": "shape_error_percent", "trend": "trend_error_percent",
        "alpha": "alpha_precision_percent", "beta": "beta_recall_percent",
        "auc": "mle_xgboost_auc", "detection": "detection_score",
        "dcr": "dcr_percent_50_ideal",
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        indexed = {norm(name): name for name in fields}
        method_column = indexed.get(norm("method"))
        columns = {key: indexed.get(norm(alias)) for key, alias in aliases.items()}
        if method_column is None or any(value is None for value in columns.values()):
            raise ValueError(f"unresolved metric columns in {path}")
        output = {}
        for row in reader:
            method = str(row[method_column]).strip()
            if not method or method in output:
                raise ValueError(f"blank/duplicate method in {path}")
            output[method] = {
                key: finite(row[column], f"{path}:{method}:{key}")
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


def load(pattern: str) -> dict[tuple[str, str], dict]:
    paths = sorted(Path(value).resolve() for value in glob.glob(pattern) if Path(value).is_dir())
    runs: dict[tuple[str, str], dict] = {}
    errors: list[str] = []
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            variant, seed = provenance.get("variant", ""), provenance.get("seed", "")
            key = (variant, seed)
            if variant not in VARIANTS or seed not in SEEDS or key in runs:
                raise ValueError(f"invalid/duplicate cell {key!r}")
            expected_mode = "uniform_search" if variant == "uniform150" else "rake_marginal_pair_select"
            expected = {
                "dataset": "default", "mode": expected_mode,
                "pool_samples": "16384", "selected_samples": "8192",
                "steps": "50", "draws": "128", "exit_code": "0",
            }
            for name, value in expected.items():
                if provenance.get(name) != value:
                    raise ValueError(f"{name}={provenance.get(name)!r}, expected {value!r}")
            metrics = read_metrics(path / "eval" / "baseline_metrics.csv")
            if set(metrics) != {"B", "G", "R", variant}:
                raise ValueError(f"unexpected methods {sorted(metrics)}")
            metadata = json.loads((path / "resampled" / "selection_metadata.json").read_text())
            if metadata.get("mode") != expected_mode or int(metadata.get("seed")) != int(seed):
                raise ValueError("metadata mode/seed mismatch")
            if metadata.get("pool_rows") != 16384 or metadata.get("selected_rows") != 8192:
                raise ValueError("pool/selection size mismatch")
            if metadata.get("no_duplicate_selected_rows") is not True:
                raise ValueError("duplicate selected indices")
            best = finite(metadata["selection"]["best"]["objective"], "best objective")
            control = finite(metadata["selection"]["uniform_control"]["objective"], "control objective")
            if best > control + 1e-12:
                raise ValueError("selection objective worse than internal control")
            runs[key] = {
                "path": str(path), "metrics": metrics, "metadata": metadata,
                "provenance": provenance, "objective_gain": control - best,
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    expected_cells = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if len(paths) != len(expected_cells) or set(runs) != expected_cells:
        errors.append(f"matched={len(paths)} missing={sorted(expected_cells - set(runs))}")
    if errors:
        raise ValueError("invalid Pareto screen:\n- " + "\n- ".join(errors))
    return runs


def assess(variant: str, runs: Mapping[tuple[str, str], dict]) -> dict:
    rows = []
    for seed in SEEDS:
        run = runs[(variant, seed)]
        metrics = run["metrics"]
        rows.append({
            "seed": seed, "candidate": metrics[variant],
            "versus_g": benefit(metrics[variant], metrics["G"]),
            "versus_r": benefit(metrics[variant], metrics["R"]),
            "versus_b": benefit(metrics[variant], metrics["B"]),
            "objective_gain": run["objective_gain"], "path": run["path"],
        })
    means = {
        reference: {metric: float(np.mean([row[reference][metric] for row in rows])) for metric in METRICS}
        for reference in ("versus_g", "versus_r", "versus_b")
    }
    wins = {
        reference: {metric: int(sum(row[reference][metric] > 0 for row in rows)) for metric in METRICS}
        for reference in ("versus_g", "versus_r")
    }
    raw = {metric: float(np.mean([row["candidate"][metric] for row in rows])) for metric in METRICS}
    reasons: list[str] = []
    for reference in ("versus_g", "versus_r"):
        label = reference.rsplit("_", 1)[-1].upper()
        for metric in ("shape", "trend"):
            if means[reference][metric] <= 0:
                reasons.append(f"mean {metric} vs {label} is not positive")
            if wins[reference][metric] < 5:
                reasons.append(f"{metric} wins vs {label} {wins[reference][metric]}/6 < 5/6")
    for metric in ("shape", "trend"):
        if means["versus_b"][metric] <= 0:
            reasons.append(f"mean {metric} vs B is not positive")
    for metric, floor in {"alpha": -.08, "beta": -.20, "auc": -.005, "dcr": -.75}.items():
        if means["versus_g"][metric] < floor:
            reasons.append(f"{metric} benefit vs G {means['versus_g'][metric]:.6f} < {floor}")
    if min(row["objective_gain"] for row in rows) <= 0:
        reasons.append("objective did not beat internal control on every seed")
    score = (.40 * means["versus_g"]["shape"] + .35 * means["versus_g"]["trend"] +
             .15 * means["versus_g"]["alpha"] + 5.0 * means["versus_g"]["auc"] +
             .05 * means["versus_g"]["beta"])
    return {
        "variant": variant, "eligible": not reasons, "score": score,
        "rejection_reasons": reasons, "mean_versus_g_oriented": means["versus_g"],
        "mean_versus_r_oriented": means["versus_r"],
        "mean_versus_b_oriented": means["versus_b"],
        "wins_versus_g": wins["versus_g"], "wins_versus_r": wins["versus_r"],
        "mean_candidate_raw": raw, "per_seed": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = load(args.input_glob)
    results = [assess(variant, runs) for variant in VARIANTS]
    results.sort(key=lambda item: (not item["eligible"], -item["score"], item["variant"]))
    promoted = [item["variant"] for item in results if item["eligible"]][:2]
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True, "seeds": list(SEEDS),
        "promoted_candidates": promoted, "variants": results,
    }
    (args.output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "selected_candidates.txt").write_text("\n".join(promoted) + ("\n" if promoted else ""))
    lines = [
        "# Default Pareto-particle multi-seed screen", "",
        f"- Promoted: **{', '.join(promoted) if promoted else 'none'}**", "",
        "| Rank | Arm | Eligible | Score | dShape/G | wins | dTrend/G | wins | dAlpha/G | dBeta/G | dAUC/G | dShape/R | dTrend/R | Mean Shape | Mean Trend |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, item in enumerate(results, start=1):
        dg, dr, wins, raw = item["mean_versus_g_oriented"], item["mean_versus_r_oriented"], item["wins_versus_g"], item["mean_candidate_raw"]
        lines.append(f"| {rank} | {item['variant']} | {'yes' if item['eligible'] else 'no'} | {item['score']:.6f} | {dg['shape']:.6f} | {wins['shape']}/6 | {dg['trend']:.6f} | {wins['trend']}/6 | {dg['alpha']:.6f} | {dg['beta']:.6f} | {dg['auc']:.6f} | {dr['shape']:.6f} | {dr['trend']:.6f} | {raw['shape']:.6f} | {raw['trend']:.6f} |")
    lines.extend(["", "## Rejections", ""])
    for item in results:
        if not item["eligible"]:
            lines.append(f"- **{item['variant']}**: " + "; ".join(item["rejection_reasons"]))
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.output / "SUMMARY.md")
    print(f"Promoted: {', '.join(promoted) or 'none'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
