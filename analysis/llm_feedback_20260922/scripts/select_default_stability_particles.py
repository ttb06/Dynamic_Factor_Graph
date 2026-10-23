#!/usr/bin/env python3
"""Fail-closed multi-seed selector for stability-aggregated particle sets."""
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


VARIANTS = ("gt4", "gtu2", "gtu4", "gttu3")
SEEDS = tuple(str(value) for value in range(20261391, 20261397))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = sorted(Path(value).resolve() for value in glob.glob(args.input_glob) if Path(value).is_dir())
    runs: dict[tuple[str, str], dict] = {}
    errors = []
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            variant, seed = provenance.get("variant", ""), provenance.get("seed", "")
            key = (variant, seed)
            if variant not in VARIANTS or seed not in SEEDS or key in runs:
                raise ValueError(f"invalid/duplicate cell {key!r}")
            expected = {
                "stage": "Default_stability_particle_screen", "dataset": "default",
                "mode": "consensus_rake", "pool_samples": "16384",
                "selected_samples": "8192", "steps": "50", "draws": "128", "exit_code": "0",
            }
            for name, value in expected.items():
                if provenance.get(name) != value:
                    raise ValueError(f"{name} mismatch")
            metrics = read_metrics(path / "eval" / "baseline_metrics.csv")
            if set(metrics) != {"B", "G", "R", variant}:
                raise ValueError(f"unexpected methods {sorted(metrics)}")
            metadata = json.loads((path / "resampled" / "selection_metadata.json").read_text())
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
            runs[key] = {"path": str(path), "metrics": metrics, "objective_gain": control - best}
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    expected_cells = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if len(paths) != len(expected_cells) or set(runs) != expected_cells:
        errors.append(f"matched={len(paths)} missing={sorted(expected_cells - set(runs))}")
    if errors:
        raise ValueError("invalid stability screen:\n- " + "\n- ".join(errors))

    results = []
    for variant in VARIANTS:
        rows = []
        for seed in SEEDS:
            metrics = runs[(variant, seed)]["metrics"]
            rows.append({
                "seed": seed, "candidate": metrics[variant],
                "versus_g": benefit(metrics[variant], metrics["G"]),
                "versus_r": benefit(metrics[variant], metrics["R"]),
                "versus_b": benefit(metrics[variant], metrics["B"]),
                "objective_gain": runs[(variant, seed)]["objective_gain"],
                "path": runs[(variant, seed)]["path"],
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
            "variant": variant, "eligible": not reasons, "score": score,
            "rejection_reasons": reasons, "mean_versus_g_oriented": means["versus_g"],
            "mean_versus_r_oriented": means["versus_r"], "mean_versus_b_oriented": means["versus_b"],
            "wins_versus_g": wins["versus_g"], "wins_versus_r": wins["versus_r"],
            "mean_candidate_raw": raw, "per_seed": rows,
        })
    results.sort(key=lambda item: (not item["eligible"], -item["score"], item["variant"]))
    promoted = [item["variant"] for item in results if item["eligible"]][:2]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps({
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True, "seeds": list(SEEDS), "promoted_candidates": promoted,
        "variants": results,
    }, indent=2) + "\n")
    (args.output / "selected_candidates.txt").write_text("\n".join(promoted) + ("\n" if promoted else ""))
    lines = [
        "# Default stability-particle multi-seed screen", "",
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
