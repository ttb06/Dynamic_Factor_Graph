#!/usr/bin/env python3
"""Fail-closed summary for fresh-pool rake_marg35 confirmation."""
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


SEEDS = tuple(str(value) for value in range(20261391, 20261397))
METHODS = {"B", "G", "R", "rake_marg35"}
METRICS = ("shape", "trend", "alpha", "beta", "auc", "detection", "dcr")
TAB_SYN = {"shape": 1.4810, "trend": 1.6499, "alpha": 99.2198, "beta": 47.7816, "auc": 0.7127}


def norm(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def resolve(names: list[str], alias: str) -> str:
    indexed = {norm(name): name for name in names}
    if norm(alias) not in indexed:
        raise ValueError(f"missing {alias!r} from {names!r}")
    return indexed[norm(alias)]


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
        names = reader.fieldnames or []
        rows = list(reader)
    method_column = resolve(names, "method")
    columns = {key: resolve(names, alias) for key, alias in aliases.items()}
    output = {
        str(row[method_column]).strip(): {
            key: finite(row[column], f"{path}:{key}") for key, column in columns.items()
        } for row in rows
    }
    if set(output) != METHODS:
        raise ValueError(f"{path}: methods={sorted(output)}")
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
    draws = data[rng.integers(0, len(data), size=(20000, len(data)))].mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = sorted(Path(value).resolve() for value in glob.glob(args.input_glob) if Path(value).is_dir())
    runs: dict[str, dict] = {}
    errors: list[str] = []
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            seed = provenance.get("seed", "")
            if seed not in SEEDS or seed in runs:
                raise ValueError(f"invalid/duplicate seed {seed!r}")
            expected = {
                "dataset": "default", "variant": "rake_marg35", "mode": "rake_marginal",
                "pool_samples": "16384", "selected_samples": "8192", "steps": "50", "exit_code": "0",
            }
            for key, value in expected.items():
                if provenance.get(key) != value:
                    raise ValueError(f"{key}={provenance.get(key)!r}, expected {value!r}")
            metadata = json.loads((path / "resampled" / "selection_metadata.json").read_text())
            if metadata.get("mode") != "rake_marginal" or int(metadata.get("seed")) != int(seed):
                raise ValueError("metadata mode/seed mismatch")
            if metadata.get("pool_rows") != 16384 or metadata.get("selected_rows") != 8192 or metadata.get("no_duplicate_selected_rows") is not True:
                raise ValueError("invalid selection metadata")
            best = finite(metadata["selection"]["best"]["objective"], "best objective")
            control = finite(metadata["selection"]["uniform_control"]["objective"], "control objective")
            if best > control + 1e-12:
                raise ValueError("moment objective worse than internal control")
            metrics = read_metrics(path / "eval" / "baseline_metrics.csv")
            runs[seed] = {
                "metrics": metrics, "path": str(path),
                "moment_gain": control - best,
                "ess": finite(metadata["selection"]["effective_sample_size"], "ESS"),
                "runtime": finite(provenance.get("pre_eval_runtime_seconds"), "runtime"),
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if len(paths) != len(SEEDS) or set(runs) != set(SEEDS):
        errors.append(f"matched={len(paths)}; missing={sorted(set(SEEDS)-set(runs))}")
    if errors:
        raise ValueError("invalid rake confirmation:\n- " + "\n- ".join(errors))

    per_seed = []
    for seed in SEEDS:
        metrics = runs[seed]["metrics"]
        per_seed.append({
            "seed": seed, "candidate": metrics["rake_marg35"],
            "versus_g": benefit(metrics["rake_marg35"], metrics["G"]),
            "versus_r": benefit(metrics["rake_marg35"], metrics["R"]),
            "versus_b": benefit(metrics["rake_marg35"], metrics["B"]),
            "moment_gain": runs[seed]["moment_gain"], "ess": runs[seed]["ess"],
            "runtime": runs[seed]["runtime"], "path": runs[seed]["path"],
        })
    means = {
        reference: {metric: float(np.mean([row[reference][metric] for row in per_seed])) for metric in METRICS}
        for reference in ("versus_g", "versus_r", "versus_b")
    }
    wins = {
        reference: {metric: int(sum(row[reference][metric] > 0 for row in per_seed)) for metric in METRICS}
        for reference in ("versus_g", "versus_r")
    }
    ci_g = {metric: bootstrap([row["versus_g"][metric] for row in per_seed], 20261420 + index) for index, metric in enumerate(METRICS)}
    raw = {metric: float(np.mean([row["candidate"][metric] for row in per_seed])) for metric in METRICS}
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
    for metric, floor in {"alpha": -.08, "beta": -.20, "auc": -.005, "dcr": -.75}.items():
        if means["versus_g"][metric] < floor:
            reasons.append(f"mean {metric} benefit vs G {means['versus_g'][metric]:.6f} < {floor}")
    if min(row["moment_gain"] for row in per_seed) <= 0:
        reasons.append("moment objective did not beat internal control on every seed")
    confirmed = not reasons
    score = .40 * means["versus_g"]["shape"] + .35 * means["versus_g"]["trend"] + .15 * means["versus_g"]["alpha"] + 5 * means["versus_g"]["auc"] + .05 * means["versus_g"]["beta"]
    payload = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True, "variant": "rake_marg35", "seeds": list(SEEDS),
        "confirmed": confirmed, "rejection_reasons": reasons, "score": score,
        "mean_versus_g_oriented": means["versus_g"], "mean_versus_r_oriented": means["versus_r"], "mean_versus_b_oriented": means["versus_b"],
        "wins_versus_g": wins["versus_g"], "wins_versus_r": wins["versus_r"],
        "bootstrap_95pct_mean_versus_g": ci_g, "mean_candidate_raw": raw,
        "mean_moment_gain": float(np.mean([row["moment_gain"] for row in per_seed])),
        "mean_effective_sample_size": float(np.mean([row["ess"] for row in per_seed])),
        "mean_pre_eval_runtime_seconds": float(np.mean([row["runtime"] for row in per_seed])),
        "published_context_only": {"TabSyn_local_one_run": TAB_SYN},
        "beats_local_tabsyn_context": {"shape": raw["shape"] < TAB_SYN["shape"], "trend": raw["trend"] < TAB_SYN["trend"]},
        "per_seed": per_seed,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "confirmation.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Default rake_marg35 fresh-pool confirmation", "",
        f"- Confirmed: **{'yes' if confirmed else 'no'}**", "",
        "| dShape/G | wins | 95% CI | dTrend/G | wins | 95% CI | dAlpha/G | dBeta/G | dAUC/G | dShape/R | dTrend/R | Mean Shape | Mean Trend |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {means['versus_g']['shape']:.6f} | {wins['versus_g']['shape']}/6 | [{ci_g['shape'][0]:.6f}, {ci_g['shape'][1]:.6f}] | {means['versus_g']['trend']:.6f} | {wins['versus_g']['trend']}/6 | [{ci_g['trend'][0]:.6f}, {ci_g['trend'][1]:.6f}] | {means['versus_g']['alpha']:.6f} | {means['versus_g']['beta']:.6f} | {means['versus_g']['auc']:.6f} | {means['versus_r']['shape']:.6f} | {means['versus_r']['trend']:.6f} | {raw['shape']:.6f} | {raw['trend']:.6f} |",
        "", "## Seed-level primary deltas", "",
        "| Seed | dShape/G | dTrend/G | dShape/R | dTrend/R | dAlpha/G | dBeta/G | dAUC/G |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in per_seed:
        dg, dr = row["versus_g"], row["versus_r"]
        lines.append(f"| {row['seed']} | {dg['shape']:.6f} | {dg['trend']:.6f} | {dr['shape']:.6f} | {dr['trend']:.6f} | {dg['alpha']:.6f} | {dg['beta']:.6f} | {dg['auc']:.6f} |")
    if reasons:
        lines.extend(["", "## Rejections", "", *[f"- {reason}" for reason in reasons]])
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.output / "SUMMARY.md")
    print(f"Confirmed: {confirmed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
