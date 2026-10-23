#!/usr/bin/env python3
"""Fail-closed six-fresh-pool confirmation summary for locked gt4."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from select_default_pareto_particles import METRICS, benefit, finite, read_env, read_metrics


SEEDS = tuple(str(value) for value in range(20261411, 20261417))


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
    errors = []
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            seed = provenance.get("seed", "")
            if seed not in SEEDS or seed in runs:
                raise ValueError(f"invalid/duplicate seed {seed!r}")
            expected = {
                "stage": "Default_stability_gt4_fresh_confirmation", "dataset": "default",
                "variant": "gt4", "mode": "consensus_rake", "pool_samples": "16384",
                "selected_samples": "8192", "steps": "50", "draws": "128",
                "consensus_strength": "4", "priors": "G:1,tail20:2", "exit_code": "0",
            }
            for name, value in expected.items():
                if provenance.get(name) != value:
                    raise ValueError(f"{name}={provenance.get(name)!r}, expected {value!r}")
            metrics = read_metrics(path / "eval" / "baseline_metrics.csv")
            if set(metrics) != {"B", "G", "R", "gt4"}:
                raise ValueError(f"unexpected methods {sorted(metrics)}")
            metadata = json.loads((path / "gt4" / "selection_metadata.json").read_text())
            if metadata.get("mode") != "consensus_rake" or int(metadata.get("seed")) != int(seed):
                raise ValueError("metadata identity mismatch")
            if metadata.get("pool_rows") != 16384 or metadata.get("selected_rows") != 8192 or metadata.get("no_duplicate_selected_rows") is not True:
                raise ValueError("invalid selected indices")
            if metadata.get("density_ratio", {}).get("classifier") != "consensus_prior":
                raise ValueError("missing consensus prior diagnostics")
            best = finite(metadata["selection"]["best"]["objective"], "best objective")
            control = finite(metadata["selection"]["uniform_control"]["objective"], "control objective")
            if best > control + 1e-12:
                raise ValueError("objective worse than internal control")
            runs[seed] = {"path": str(path), "metrics": metrics, "objective_gain": control - best}
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if len(paths) != 6 or set(runs) != set(SEEDS):
        errors.append(f"matched={len(paths)} missing={sorted(set(SEEDS)-set(runs))}")
    if errors:
        raise ValueError("invalid gt4 confirmation:\n- " + "\n- ".join(errors))

    rows = []
    for seed in SEEDS:
        metrics = runs[seed]["metrics"]
        rows.append({
            "seed": seed, "candidate": metrics["gt4"],
            "versus_g": benefit(metrics["gt4"], metrics["G"]),
            "versus_r": benefit(metrics["gt4"], metrics["R"]),
            "versus_b": benefit(metrics["gt4"], metrics["B"]),
            "objective_gain": runs[seed]["objective_gain"], "path": runs[seed]["path"],
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
    ci = {metric: bootstrap([row["versus_g"][metric] for row in rows], 20261430 + index) for index, metric in enumerate(METRICS)}
    reasons = []
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
    confirmed = not reasons
    score = (.40 * means["versus_g"]["shape"] + .35 * means["versus_g"]["trend"] +
             .15 * means["versus_g"]["alpha"] + 5.0 * means["versus_g"]["auc"] +
             .05 * means["versus_g"]["beta"])
    payload = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True, "variant": "gt4", "seeds": list(SEEDS),
        "confirmed": confirmed, "rejection_reasons": reasons, "score": score,
        "mean_versus_g_oriented": means["versus_g"], "mean_versus_r_oriented": means["versus_r"],
        "mean_versus_b_oriented": means["versus_b"], "wins_versus_g": wins["versus_g"],
        "wins_versus_r": wins["versus_r"], "bootstrap_95pct_mean_versus_g": ci,
        "mean_candidate_raw": raw, "per_seed": rows,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "confirmation.json").write_text(json.dumps(payload, indent=2) + "\n")
    dg, dr = means["versus_g"], means["versus_r"]
    lines = [
        "# Default gt4 fresh-pool confirmation", "", f"- Confirmed: **{'yes' if confirmed else 'no'}**", "",
        "| dShape/G | wins | 95% CI | dTrend/G | wins | 95% CI | dAlpha/G | dBeta/G | dAUC/G | dShape/R | dTrend/R | Mean Shape | Mean Trend |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {dg['shape']:.6f} | {wins['versus_g']['shape']}/6 | [{ci['shape'][0]:.6f}, {ci['shape'][1]:.6f}] | {dg['trend']:.6f} | {wins['versus_g']['trend']}/6 | [{ci['trend'][0]:.6f}, {ci['trend'][1]:.6f}] | {dg['alpha']:.6f} | {dg['beta']:.6f} | {dg['auc']:.6f} | {dr['shape']:.6f} | {dr['trend']:.6f} | {raw['shape']:.6f} | {raw['trend']:.6f} |",
        "", "## Seed-level primary deltas", "",
        "| Seed | dShape/G | dTrend/G | dShape/R | dTrend/R | dBeta/G | dAUC/G |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        dg_row, dr_row = row["versus_g"], row["versus_r"]
        lines.append(f"| {row['seed']} | {dg_row['shape']:.6f} | {dg_row['trend']:.6f} | {dr_row['shape']:.6f} | {dr_row['trend']:.6f} | {dg_row['beta']:.6f} | {dg_row['auc']:.6f} |")
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
