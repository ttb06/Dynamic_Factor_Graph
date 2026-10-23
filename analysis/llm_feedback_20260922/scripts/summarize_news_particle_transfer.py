#!/usr/bin/env python3
"""Fail-closed two-seed summary for zero-shot News particle transfer."""
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


SEEDS = ("20261431", "20261432")
VARIANTS = ("gt4", "int20_t005")
METRICS = ("shape", "trend", "alpha", "beta", "r2", "nrmse", "violation")


def finite(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: nonnumeric {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context}: nonfinite {value!r}")
    return result


def read_env(path: Path) -> dict[str, str]:
    output = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            output[key.strip()] = value.strip()
    return output


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    fields = {
        "shape": "shape_score_percent", "trend": "trend_score_percent",
        "alpha": "alpha_precision_percent", "beta": "beta_recall_percent",
        "r2": "xgboost_regression_r2", "nrmse": "xgboost_regression_nrmse",
        "violation": "any_constraint_violation_percent",
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    output = {}
    for row in rows:
        method = str(row["method"]).strip()
        if not method or method in output:
            raise ValueError(f"blank/duplicate method in {path}")
        output[method] = {name: finite(row[column], f"{method}:{name}") for name, column in fields.items()}
    return output


def benefit(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {
        "shape": left["shape"] - right["shape"],
        "trend": left["trend"] - right["trend"],
        "alpha": left["alpha"] - right["alpha"],
        "beta": left["beta"] - right["beta"],
        "r2": left["r2"] - right["r2"],
        "nrmse": right["nrmse"] - left["nrmse"],
        "violation": right["violation"] - left["violation"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = sorted(Path(value).resolve() for value in glob.glob(args.input_glob) if Path(value).is_dir())
    runs = {}
    errors = []
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            seed = provenance.get("seed", "")
            if seed not in SEEDS or seed in runs:
                raise ValueError(f"invalid/duplicate seed {seed!r}")
            expected = {
                "stage": "News_zero_shot_particle_transfer", "dataset": "news",
                "pool_samples": "16384", "selected_samples": "8192", "steps": "50",
                "variants": "gt4,int20_t005", "exit_code": "0",
            }
            for name, value in expected.items():
                if provenance.get(name) != value:
                    raise ValueError(f"{name} mismatch")
            metrics = read_metrics(path / "eval" / "metrics.csv")
            if set(metrics) != {"B", "G", "R", *VARIANTS}:
                raise ValueError(f"unexpected methods {sorted(metrics)}")
            metadata = {}
            for variant in VARIANTS:
                item = json.loads((path / variant / "selection_metadata.json").read_text())
                if item.get("dataset") != "news" or int(item.get("seed")) != int(seed):
                    raise ValueError(f"{variant}: metadata mismatch")
                if item.get("pool_rows") != 16384 or item.get("selected_rows") != 8192 or item.get("no_duplicate_selected_rows") is not True:
                    raise ValueError(f"{variant}: invalid selection")
                if variant == "int20_t005":
                    integrated = item.get("integrated_coverage", {})
                    if integrated.get("enabled") is not True or finite(integrated.get("selected_deficit"), "deficit") > 1e-12:
                        raise ValueError("int20_t005 missed train barrier")
                metadata[variant] = item
            runs[seed] = {"path": str(path), "metrics": metrics, "metadata": metadata}
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if len(paths) != 2 or set(runs) != set(SEEDS):
        errors.append(f"matched={len(paths)} missing={sorted(set(SEEDS)-set(runs))}")
    if errors:
        raise ValueError("invalid News transfer:\n- " + "\n- ".join(errors))

    results = []
    for variant in VARIANTS:
        rows = []
        for seed in SEEDS:
            metrics = runs[seed]["metrics"]
            rows.append({
                "seed": seed, "candidate": metrics[variant],
                "versus_g": benefit(metrics[variant], metrics["G"]),
                "versus_r": benefit(metrics[variant], metrics["R"]),
                "versus_b": benefit(metrics[variant], metrics["B"]),
                "path": runs[seed]["path"],
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
        reasons = []
        for reference in ("versus_g", "versus_r"):
            comparator = reference.rsplit("_", 1)[-1].upper()
            for metric in ("shape", "trend"):
                if means[reference][metric] <= 0:
                    reasons.append(f"mean {metric} vs {comparator} is not positive")
                if wins[reference][metric] < 2:
                    reasons.append(f"{metric} wins vs {comparator} {wins[reference][metric]}/2 < 2/2")
        for metric in ("shape", "trend"):
            if means["versus_b"][metric] <= 0:
                reasons.append(f"mean {metric} vs B is not positive")
        for metric, floor in {"alpha": -.10, "beta": -.20, "r2": -.10, "nrmse": -.05}.items():
            if means["versus_g"][metric] < floor:
                reasons.append(f"{metric} benefit vs G {means['versus_g'][metric]:.6f} < {floor}")
        score = (.35 * means["versus_g"]["shape"] + .30 * means["versus_g"]["trend"] +
                 .10 * means["versus_g"]["alpha"] + .05 * means["versus_g"]["beta"] +
                 .15 * means["versus_g"]["r2"] + .05 * means["versus_g"]["nrmse"])
        results.append({
            "variant": variant, "eligible": not reasons, "score": score,
            "rejection_reasons": reasons, "mean_versus_g_oriented": means["versus_g"],
            "mean_versus_r_oriented": means["versus_r"], "mean_versus_b_oriented": means["versus_b"],
            "wins_versus_g": wins["versus_g"], "wins_versus_r": wins["versus_r"],
            "mean_candidate_raw": raw, "per_seed": rows,
        })
    results.sort(key=lambda item: (not item["eligible"], -item["score"], item["variant"]))
    promoted = [item["variant"] for item in results if item["eligible"]]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps({
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True, "seeds": list(SEEDS), "promoted_candidates": promoted,
        "variants": results,
    }, indent=2) + "\n")
    lines = [
        "# News zero-shot particle transfer", "", f"- Promoted: **{', '.join(promoted) if promoted else 'none'}**", "",
        "| Rank | Arm | Eligible | dShape/G | wins | dTrend/G | wins | dAlpha/G | dBeta/G | dR2/G | dNRMSE+/G | dShape/R | dTrend/R |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, item in enumerate(results, start=1):
        dg, dr, wins = item["mean_versus_g_oriented"], item["mean_versus_r_oriented"], item["wins_versus_g"]
        lines.append(f"| {rank} | {item['variant']} | {'yes' if item['eligible'] else 'no'} | {dg['shape']:.6f} | {wins['shape']}/2 | {dg['trend']:.6f} | {wins['trend']}/2 | {dg['alpha']:.6f} | {dg['beta']:.6f} | {dg['r2']:.6f} | {dg['nrmse']:.6f} | {dr['shape']:.6f} | {dr['trend']:.6f} |")
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
