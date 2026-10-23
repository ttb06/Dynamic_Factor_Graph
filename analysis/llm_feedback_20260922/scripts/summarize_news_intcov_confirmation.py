#!/usr/bin/env python3
"""Primary and comprehensive six-seed confirmation for News int20_t005."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from summarize_news_particle_transfer import METRICS, benefit, finite, read_env, read_metrics


SEEDS = tuple(str(value) for value in range(20261441, 20261447))


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
    runs = {}
    errors = []
    for path in paths:
        try:
            provenance = read_env(path / "provenance.txt")
            seed = provenance.get("seed", "")
            if seed not in SEEDS or seed in runs:
                raise ValueError(f"invalid/duplicate seed {seed!r}")
            expected = {
                "stage": "News_int20_t005_fresh_confirmation", "dataset": "news",
                "variant": "int20_t005", "pool_samples": "16384", "selected_samples": "8192",
                "steps": "50", "draws": "128", "consensus_strength": "4",
                "priors": "G:1,tail20:2", "geometric_coverage_weight": "20",
                "geometric_coverage_tolerance": "0", "integrated_coverage_weight": "20",
                "integrated_coverage_tolerance": ".005", "integrated_coverage_shortlist": "8",
                "integrated_coverage_folds": "2", "exit_code": "0",
            }
            for name, value in expected.items():
                if provenance.get(name) != value:
                    raise ValueError(f"{name} mismatch")
            metrics = read_metrics(path / "eval" / "metrics.csv")
            if set(metrics) != {"B", "G", "R", "int20_t005"}:
                raise ValueError(f"unexpected methods {sorted(metrics)}")
            metadata = json.loads((path / "int20_t005" / "selection_metadata.json").read_text())
            integrated = metadata.get("integrated_coverage", {})
            if (metadata.get("dataset") != "news" or int(metadata.get("seed")) != int(seed) or
                    metadata.get("no_duplicate_selected_rows") is not True or
                    integrated.get("enabled") is not True):
                raise ValueError("invalid candidate metadata")
            runs[seed] = {
                "path": str(path), "metrics": metrics,
                "integrated_deficit": finite(integrated.get("selected_deficit"), "deficit"),
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if len(paths) != 6 or set(runs) != set(SEEDS):
        errors.append(f"matched={len(paths)} missing={sorted(set(SEEDS)-set(runs))}")
    if errors:
        raise ValueError("invalid News confirmation:\n- " + "\n- ".join(errors))

    rows = []
    for seed in SEEDS:
        metrics = runs[seed]["metrics"]
        rows.append({
            "seed": seed, "candidate": metrics["int20_t005"],
            "versus_g": benefit(metrics["int20_t005"], metrics["G"]),
            "versus_r": benefit(metrics["int20_t005"], metrics["R"]),
            "versus_b": benefit(metrics["int20_t005"], metrics["B"]),
            "integrated_deficit": runs[seed]["integrated_deficit"], "path": runs[seed]["path"],
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
    ci = {metric: bootstrap([row["versus_g"][metric] for row in rows], 20261470 + index) for index, metric in enumerate(METRICS)}
    primary_reasons = []
    for reference in ("versus_g", "versus_r"):
        comparator = reference.rsplit("_", 1)[-1].upper()
        for metric in ("shape", "trend"):
            if means[reference][metric] <= 0:
                primary_reasons.append(f"mean {metric} vs {comparator} is not positive")
            if wins[reference][metric] < 5:
                primary_reasons.append(f"{metric} wins vs {comparator} {wins[reference][metric]}/6 < 5/6")
    for metric in ("shape", "trend"):
        if means["versus_b"][metric] <= 0:
            primary_reasons.append(f"mean {metric} vs B is not positive")
        if ci[metric][0] <= 0:
            primary_reasons.append(f"{metric} bootstrap interval crosses zero")
    if max(row["integrated_deficit"] for row in rows) > 1e-12:
        primary_reasons.append("two-fold train coverage barrier missed on at least one seed")
    comprehensive_reasons = list(primary_reasons)
    for metric, floor in {"alpha": -.10, "beta": -.20, "r2": -.10, "nrmse": -.05}.items():
        if means["versus_g"][metric] < floor:
            comprehensive_reasons.append(f"{metric} benefit vs G {means['versus_g'][metric]:.6f} < {floor}")
    primary_confirmed = not primary_reasons
    comprehensive_confirmed = not comprehensive_reasons
    payload = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True, "variant": "int20_t005", "seeds": list(SEEDS),
        "primary_metric_confirmed": primary_confirmed,
        "comprehensive_confirmed": comprehensive_confirmed,
        "primary_rejection_reasons": primary_reasons,
        "comprehensive_rejection_reasons": comprehensive_reasons,
        "mean_versus_g_oriented": means["versus_g"], "mean_versus_r_oriented": means["versus_r"],
        "mean_versus_b_oriented": means["versus_b"], "wins_versus_g": wins["versus_g"],
        "wins_versus_r": wins["versus_r"], "bootstrap_95pct_mean_versus_g": ci,
        "mean_candidate_raw": raw, "per_seed": rows,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "confirmation.json").write_text(json.dumps(payload, indent=2) + "\n")
    dg, dr = means["versus_g"], means["versus_r"]
    lines = [
        "# News int20_t005 six-seed confirmation", "",
        f"- Primary-metric confirmed: **{'yes' if primary_confirmed else 'no'}**",
        f"- Comprehensive confirmed: **{'yes' if comprehensive_confirmed else 'no'}**", "",
        "| dShape/G | wins | 95% CI | dTrend/G | wins | 95% CI | dAlpha/G | dBeta/G | dR2/G | dNRMSE+/G | dShape/R | dTrend/R | Mean Shape | Mean Trend |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {dg['shape']:.6f} | {wins['versus_g']['shape']}/6 | [{ci['shape'][0]:.6f}, {ci['shape'][1]:.6f}] | {dg['trend']:.6f} | {wins['versus_g']['trend']}/6 | [{ci['trend'][0]:.6f}, {ci['trend'][1]:.6f}] | {dg['alpha']:.6f} | {dg['beta']:.6f} | {dg['r2']:.6f} | {dg['nrmse']:.6f} | {dr['shape']:.6f} | {dr['trend']:.6f} | {raw['shape']:.6f} | {raw['trend']:.6f} |",
        "", "## Seed-level primary deltas", "",
        "| Seed | dShape/G | dTrend/G | dShape/R | dTrend/R | dAlpha/G | dBeta/G | dR2/G |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        dg_row, dr_row = row["versus_g"], row["versus_r"]
        lines.append(f"| {row['seed']} | {dg_row['shape']:.6f} | {dg_row['trend']:.6f} | {dr_row['shape']:.6f} | {dr_row['trend']:.6f} | {dg_row['alpha']:.6f} | {dg_row['beta']:.6f} | {dg_row['r2']:.6f} |")
    if comprehensive_reasons:
        lines.extend(["", "## Comprehensive rejections", "", *[f"- {reason}" for reason in comprehensive_reasons]])
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.output / "SUMMARY.md")
    print(f"Primary={primary_confirmed}; comprehensive={comprehensive_confirmed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
