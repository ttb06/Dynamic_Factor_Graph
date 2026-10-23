#!/usr/bin/env python3
"""Fail-closed six-seed summary for balanced-particle confirmation."""
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


VARIANTS = ("ratio_xgb50", "hybrid_logit50")
MODES = {"ratio_xgb50": "ratio_xgb", "hybrid_logit50": "hybrid_logit"}
SEEDS = tuple(str(value) for value in range(20261381, 20261387))
POOL_SAMPLES = 16384
SELECTED_SAMPLES = 8192
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


def bootstrap(values: Sequence[float], seed: int) -> list[float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), size=(20000, len(data)))].mean(axis=1)
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def load(pattern: str) -> dict[tuple[str, str], dict[str, Any]]:
    paths = sorted(Path(value).resolve() for value in glob.glob(pattern) if Path(value).is_dir())
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            variant, seed = provenance.get("variant", ""), provenance.get("seed", "")
            key = (variant, seed)
            if variant not in VARIANTS or seed not in SEEDS or key in runs:
                raise ValueError(f"invalid/duplicate design cell {key!r}")
            if provenance.get("dataset") != "default" or provenance.get("mode") != MODES[variant]:
                raise ValueError("dataset/mode mismatch")
            if provenance.get("pool_samples") != str(POOL_SAMPLES) or provenance.get("selected_samples") != str(SELECTED_SAMPLES) or provenance.get("steps") != STEPS:
                raise ValueError("design mismatch")
            if provenance.get("exit_code") != "0":
                raise ValueError("missing successful exit marker")
            metrics = read_metrics(path / "eval" / "baseline_metrics.csv")
            if set(metrics) != {"B", "G", "R", variant}:
                raise ValueError(f"unexpected methods {sorted(metrics)}")
            metadata = json.loads((path / "resampled" / "selection_metadata.json").read_text())
            if metadata.get("mode") != MODES[variant] or int(metadata.get("seed")) != int(seed):
                raise ValueError("resampling metadata mismatch")
            if metadata.get("pool_rows") != POOL_SAMPLES or metadata.get("selected_rows") != SELECTED_SAMPLES or metadata.get("no_duplicate_selected_rows") is not True:
                raise ValueError("invalid pool/selection metadata")
            best = finite(metadata["selection"]["best"]["objective"], "best objective")
            control = finite(metadata["selection"]["uniform_control"]["objective"], "control objective")
            if best > control + 1e-12:
                raise ValueError("moment objective worse than internal control")
            runs[key] = {"path": str(path), "provenance": provenance, "metrics": metrics, "metadata": metadata}
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if len(paths) != len(expected) or set(runs) != expected:
        errors.append(f"matched={len(paths)}; missing={sorted(expected-set(runs))}; extra={sorted(set(runs)-expected)}")
    if errors:
        raise ValueError("invalid particle confirmation:\n- " + "\n- ".join(errors))
    return runs


def assess(variant: str, runs: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        run = runs[(variant, seed)]
        metrics = run["metrics"]
        metadata = run["metadata"]
        per_seed.append({
            "seed": seed, "candidate": metrics[variant], "g": metrics["G"], "r": metrics["R"],
            "versus_g": benefit(metrics[variant], metrics["G"]),
            "versus_r": benefit(metrics[variant], metrics["R"]),
            "versus_b": benefit(metrics[variant], metrics["B"]),
            "moment_gain": finite(metadata["selection"]["uniform_control"]["objective"], "control") - finite(metadata["selection"]["best"]["objective"], "best"),
            "effective_sample_size": finite(metadata["selection"]["effective_sample_size"], "ESS"),
            "pre_eval_runtime_seconds": finite(run["provenance"].get("pre_eval_runtime_seconds"), "runtime"),
            "path": run["path"],
        })
    keys = ("shape", "trend", "alpha", "beta", "auc", "detection", "dcr")
    means = {
        reference: {key: float(np.mean([row[reference][key] for row in per_seed])) for key in keys}
        for reference in ("versus_g", "versus_r", "versus_b")
    }
    wins = {
        reference: {key: int(sum(row[reference][key] > 0 for row in per_seed)) for key in keys}
        for reference in ("versus_g", "versus_r")
    }
    ci_g = {key: bootstrap([row["versus_g"][key] for row in per_seed], 20261400 + index) for index, key in enumerate(keys)}
    raw = {key: float(np.mean([row["candidate"][key] for row in per_seed])) for key in keys}
    reasons: list[str] = []
    for reference in ("versus_g", "versus_r"):
        label = reference.rsplit("_", 1)[-1].upper()
        for metric in ("shape", "trend"):
            if means[reference][metric] <= 0:
                reasons.append(f"mean {metric} benefit vs {label} {means[reference][metric]:.6f} is not positive")
            if wins[reference][metric] < 5:
                reasons.append(f"{metric} wins vs {label} {wins[reference][metric]}/6 < 5/6")
    for metric in ("shape", "trend"):
        if means["versus_b"][metric] <= 0:
            reasons.append(f"mean {metric} benefit vs B {means['versus_b'][metric]:.6f} is not positive")
    floors = {"alpha": -0.08, "beta": -0.20, "auc": -0.005, "dcr": -0.75}
    for metric, floor in floors.items():
        if means["versus_g"][metric] < floor:
            reasons.append(f"mean {metric} benefit vs G {means['versus_g'][metric]:.6f} < {floor}")
    if min(row["moment_gain"] for row in per_seed) <= 0:
        reasons.append("moment objective did not beat internal control on every seed")
    score = 0.40 * means["versus_g"]["shape"] + 0.35 * means["versus_g"]["trend"] + 0.15 * means["versus_g"]["alpha"] + 5.0 * means["versus_g"]["auc"] + 0.05 * means["versus_g"]["beta"]
    return {
        "variant": variant, "confirmed": not reasons, "rejection_reasons": reasons, "score": score,
        "mean_versus_g_oriented": means["versus_g"], "mean_versus_r_oriented": means["versus_r"], "mean_versus_b_oriented": means["versus_b"],
        "wins_versus_g": wins["versus_g"], "wins_versus_r": wins["versus_r"],
        "bootstrap_95pct_mean_versus_g": ci_g, "mean_candidate_raw": raw,
        "mean_moment_gain": float(np.mean([row["moment_gain"] for row in per_seed])),
        "mean_effective_sample_size": float(np.mean([row["effective_sample_size"] for row in per_seed])),
        "mean_pre_eval_runtime_seconds": float(np.mean([row["pre_eval_runtime_seconds"] for row in per_seed])),
        "beats_local_tabsyn_context": {"shape": raw["shape"] < TAB_SYN["shape"], "trend": raw["trend"] < TAB_SYN["trend"]},
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
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True, "seeds": list(SEEDS),
        "published_context_only": {"TabSyn_local_one_run": TAB_SYN},
        "confirmed_variants": confirmed, "recommended_variant": winner, "variants": variants,
    }
    (args.output / "confirmation.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Default balanced-particle six-seed confirmation", "",
        f"- Confirmed: **{', '.join(confirmed) if confirmed else 'none'}**",
        f"- Recommended: **{winner or 'none'}**", "",
        "| Rank | Arm | Confirmed | Score | dShape/G | wins | 95% CI | dTrend/G | wins | 95% CI | dAlpha/G | dBeta/G | dAUC/G | dShape/R | dTrend/R | Mean Shape | Mean Trend |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, item in enumerate(variants, start=1):
        dg, dr = item["mean_versus_g_oriented"], item["mean_versus_r_oriented"]
        wins, ci, raw = item["wins_versus_g"], item["bootstrap_95pct_mean_versus_g"], item["mean_candidate_raw"]
        lines.append(f"| {rank} | {item['variant']} | {'yes' if item['confirmed'] else 'no'} | {item['score']:.6f} | {dg['shape']:.6f} | {wins['shape']}/6 | [{ci['shape'][0]:.6f}, {ci['shape'][1]:.6f}] | {dg['trend']:.6f} | {wins['trend']}/6 | [{ci['trend'][0]:.6f}, {ci['trend'][1]:.6f}] | {dg['alpha']:.6f} | {dg['beta']:.6f} | {dg['auc']:.6f} | {dr['shape']:.6f} | {dr['trend']:.6f} | {raw['shape']:.6f} | {raw['trend']:.6f} |")
    lines.extend(["", "## Seed-level primary deltas", "", "| Arm | Seed | dShape/G | dTrend/G | dShape/R | dTrend/R | dAlpha/G | dBeta/G | dAUC/G |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for item in variants:
        for row in item["per_seed"]:
            dg, dr = row["versus_g"], row["versus_r"]
            lines.append(f"| {item['variant']} | {row['seed']} | {dg['shape']:.6f} | {dg['trend']:.6f} | {dr['shape']:.6f} | {dr['trend']:.6f} | {dg['alpha']:.6f} | {dg['beta']:.6f} | {dg['auc']:.6f} |")
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
