#!/usr/bin/env python3
"""Fail-closed fresh-pool confirmation for selected Pareto particle arms."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from select_default_pareto_particles import (
    METRICS, VARIANTS, benefit, finite, read_env, read_metrics,
)


SEEDS = tuple(str(value) for value in range(20261401, 20261407))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--screen-job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    selected = tuple(line.strip() for line in args.selection_file.read_text().splitlines() if line.strip())
    if not 1 <= len(selected) <= 2 or len(set(selected)) != len(selected):
        raise ValueError(f"invalid selected arms: {selected!r}")
    if any(value not in VARIANTS for value in selected):
        raise ValueError(f"unknown selected arm: {selected!r}")

    paths = sorted(Path(value).resolve() for value in glob.glob(args.input_glob) if Path(value).is_dir())
    runs: dict[str, dict] = {}
    errors: list[str] = []
    expected_csv = ",".join(selected)
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            seed = provenance.get("seed", "")
            if seed not in SEEDS or seed in runs:
                raise ValueError(f"invalid/duplicate seed {seed!r}")
            expected = {
                "stage": "Default_pareto_particle_fresh_confirmation",
                "screen_job": args.screen_job, "dataset": "default",
                "selected_variants": expected_csv, "pool_samples": "16384",
                "selected_samples": "8192", "steps": "50", "draws": "128",
                "exit_code": "0",
            }
            for name, value in expected.items():
                if provenance.get(name) != value:
                    raise ValueError(f"{name}={provenance.get(name)!r}, expected {value!r}")
            metrics = read_metrics(path / "eval" / "baseline_metrics.csv")
            if set(metrics) != {"B", "G", "R", *selected}:
                raise ValueError(f"unexpected methods {sorted(metrics)}")
            metadata = {}
            objective_gain = {}
            for variant in selected:
                item = json.loads((path / variant / "selection_metadata.json").read_text())
                if int(item.get("seed")) != int(seed) or item.get("pool_rows") != 16384 or item.get("selected_rows") != 8192:
                    raise ValueError(f"{variant}: invalid metadata identity")
                if item.get("no_duplicate_selected_rows") is not True:
                    raise ValueError(f"{variant}: duplicate selected indices")
                best = finite(item["selection"]["best"]["objective"], "best objective")
                control = finite(item["selection"]["uniform_control"]["objective"], "control objective")
                if best > control + 1e-12:
                    raise ValueError(f"{variant}: objective worse than internal control")
                metadata[variant] = item
                objective_gain[variant] = control - best
            runs[seed] = {"path": str(path), "metrics": metrics, "metadata": metadata, "objective_gain": objective_gain}
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if len(paths) != 6 or set(runs) != set(SEEDS):
        errors.append(f"matched={len(paths)} missing={sorted(set(SEEDS) - set(runs))}")
    if errors:
        raise ValueError("invalid fresh confirmation:\n- " + "\n- ".join(errors))

    results = []
    for variant in selected:
        rows = []
        for seed in SEEDS:
            metrics = runs[seed]["metrics"]
            rows.append({
                "seed": seed, "candidate": metrics[variant],
                "versus_g": benefit(metrics[variant], metrics["G"]),
                "versus_r": benefit(metrics[variant], metrics["R"]),
                "versus_b": benefit(metrics[variant], metrics["B"]),
                "objective_gain": runs[seed]["objective_gain"][variant],
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
        results.append({
            "variant": variant, "confirmed": not reasons, "score": score,
            "rejection_reasons": reasons, "mean_versus_g_oriented": means["versus_g"],
            "mean_versus_r_oriented": means["versus_r"],
            "mean_versus_b_oriented": means["versus_b"],
            "wins_versus_g": wins["versus_g"], "wins_versus_r": wins["versus_r"],
            "mean_candidate_raw": raw, "per_seed": rows,
        })
    results.sort(key=lambda item: (not item["confirmed"], -item["score"], item["variant"]))
    confirmed = [item["variant"] for item in results if item["confirmed"]]
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True, "screen_job": args.screen_job,
        "seeds": list(SEEDS), "confirmed_variants": confirmed, "variants": results,
    }
    (args.output / "confirmation.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Default Pareto-particle fresh-pool confirmation", "",
        f"- Confirmed: **{', '.join(confirmed) if confirmed else 'none'}**", "",
        "| Rank | Arm | Confirmed | Score | dShape/G | wins | dTrend/G | wins | dAlpha/G | dBeta/G | dAUC/G | dShape/R | dTrend/R | Mean Shape | Mean Trend |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, item in enumerate(results, start=1):
        dg, dr, wins, raw = item["mean_versus_g_oriented"], item["mean_versus_r_oriented"], item["wins_versus_g"], item["mean_candidate_raw"]
        lines.append(f"| {rank} | {item['variant']} | {'yes' if item['confirmed'] else 'no'} | {item['score']:.6f} | {dg['shape']:.6f} | {wins['shape']}/6 | {dg['trend']:.6f} | {wins['trend']}/6 | {dg['alpha']:.6f} | {dg['beta']:.6f} | {dg['auc']:.6f} | {dr['shape']:.6f} | {dr['trend']:.6f} | {raw['shape']:.6f} | {raw['trend']:.6f} |")
    lines.extend(["", "## Seed-level primary deltas", "", "| Arm | Seed | dShape/G | dTrend/G | dShape/R | dTrend/R | dBeta/G | dAUC/G |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for item in results:
        for row in item["per_seed"]:
            dg, dr = row["versus_g"], row["versus_r"]
            lines.append(f"| {item['variant']} | {row['seed']} | {dg['shape']:.6f} | {dg['trend']:.6f} | {dr['shape']:.6f} | {dr['trend']:.6f} | {dg['beta']:.6f} | {dg['auc']:.6f} |")
    lines.extend(["", "## Rejections", ""])
    for item in results:
        if not item["confirmed"]:
            lines.append(f"- **{item['variant']}**: " + "; ".join(item["rejection_reasons"]))
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.output / "SUMMARY.md")
    print(f"Confirmed: {', '.join(confirmed) or 'none'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
