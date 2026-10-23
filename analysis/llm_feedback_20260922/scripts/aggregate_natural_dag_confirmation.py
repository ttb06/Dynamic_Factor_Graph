#!/usr/bin/env python3
"""Aggregate paired multi-seed natural-DAG controls without exposing real rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "macro_violation", "heldout_violation_error", "calibration_violation_error",
    "numeric_ks", "categorical_tv", "action_events", "unique_rows_actuated",
]


def variant(metadata: dict) -> str:
    if metadata["topology_mode"] == "frozen":
        return "frozen"
    if float(metadata["active_max_t"]) <= .11:
        return "late_only"
    return "dynamic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    for run in args.runs:
        metadata = json.loads((run / "run_metadata.json").read_text())
        frame = pd.read_csv(run / "method_metrics.csv")
        online = frame.loc[frame.method.eq("xstate_reward_calibrated")].iloc[0]
        final = frame.loc[frame.method.eq("final_reward_calibrated")].iloc[0]
        item = {"seed": int(metadata["seed"]), "variant": variant(metadata), "run": str(run)}
        item.update({metric: float(online[metric]) for metric in METRICS})
        rows.append(item)
        if item["variant"] == "dynamic":
            final_item = {"seed": item["seed"], "variant": "final_only", "run": str(run)}
            final_item.update({metric: float(final[metric]) for metric in METRICS})
            rows.append(final_item)
    raw = pd.DataFrame(rows).sort_values(["seed", "variant"])
    raw.to_csv(args.output / "per_seed_metrics.csv", index=False)

    summary_rows = []
    for name, group in raw.groupby("variant", sort=False):
        row = {"variant": name, "seeds": group.seed.nunique()}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    wide = raw.pivot(index="seed", columns="variant", values=METRICS)
    gains = []
    for seed in sorted(raw.seed.unique()):
        for baseline in ("frozen", "late_only", "final_only"):
            row = {"seed": int(seed), "baseline": baseline}
            for metric in ("macro_violation", "heldout_violation_error", "calibration_violation_error"):
                dynamic = float(wide.loc[seed, (metric, "dynamic")])
                reference = float(wide.loc[seed, (metric, baseline)])
                row[f"{metric}_gain_pct"] = 100.0 * (reference - dynamic) / max(reference, 1e-12)
            gains.append(row)
    gain_frame = pd.DataFrame(gains)
    gain_frame.to_csv(args.output / "relative_gains.csv", index=False)

    rng = np.random.default_rng(20261010)
    bootstrap_rows = []
    for baseline in ("frozen", "late_only", "final_only"):
        for metric in ("macro_violation", "heldout_violation_error", "calibration_violation_error"):
            differences = np.asarray([
                float(wide.loc[seed, (metric, baseline)]) - float(wide.loc[seed, (metric, "dynamic")])
                for seed in sorted(raw.seed.unique())
            ])
            draws = rng.choice(differences, size=(20000, len(differences)), replace=True).mean(axis=1)
            bootstrap_rows.append({
                "baseline": baseline, "metric": metric,
                "mean_absolute_reduction": float(differences.mean()),
                "bootstrap_seed_ci_low": float(np.quantile(draws, .025)),
                "bootstrap_seed_ci_high": float(np.quantile(draws, .975)),
                "wins": int((differences > 0).sum()), "seeds": len(differences),
            })
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output / "paired_seed_bootstrap.csv", index=False)

    lines = ["# Natural compositional DAG: confirmation", "",
             "All graph paths and calibration policies were frozen before these confirmation seeds.", "",
             "| Variant | Seeds | Macro violation | Held-out error | Calibration error | Numeric KS | Categorical TV | Actions |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.seeds} | {row.macro_violation_mean:.4f} ± {row.macro_violation_std:.4f} | "
            f"{row.heldout_violation_error_mean:.4f} ± {row.heldout_violation_error_std:.4f} | "
            f"{row.calibration_violation_error_mean:.4f} ± {row.calibration_violation_error_std:.4f} | "
            f"{row.numeric_ks_mean:.4f} | {row.categorical_tv_mean:.4f} | {row.action_events_mean:.1f} |"
        )
    lines += ["", "| Baseline | Macro gain | Held-out gain | Calibration gain | Seeds won on held-out |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for baseline, group in gain_frame.groupby("baseline", sort=False):
        lines.append(
            f"| {baseline} | {group.macro_violation_gain_pct.mean():.1f}% | "
            f"{group.heldout_violation_error_gain_pct.mean():.1f}% | "
            f"{group.calibration_violation_error_gain_pct.mean():.1f}% | "
            f"{int((group.heldout_violation_error_gain_pct > 0).sum())}/{len(group)} |"
        )
    lines += ["", "Paired seed-bootstrap 95% intervals below quantify sampling-seed uncertainty only; "
              "they do not cover a second trained TabDiff checkpoint.", "",
              "| Baseline | Held-out absolute reduction | 95% seed-bootstrap interval |",
              "| --- | ---: | ---: |"]
    for baseline in ("frozen", "late_only", "final_only"):
        row = bootstrap[(bootstrap.baseline == baseline) &
                        (bootstrap.metric == "heldout_violation_error")].iloc[0]
        lines.append(
            f"| {baseline} | {row.mean_absolute_reduction:.4f} | "
            f"[{row.bootstrap_seed_ci_low:.4f}, {row.bootstrap_seed_ci_high:.4f}] |"
        )
    heldout_ok = all(
        float(group.heldout_violation_error_gain_pct.mean()) >= 10.0
        for baseline, group in gain_frame.groupby("baseline") if baseline in {"frozen", "final_only"}
    )
    lines += ["", f"Predefined 10% held-out gain threshold versus frozen and final-only: **{'passed' if heldout_ok else 'not passed'}**."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "run_summary.json").write_text(json.dumps({
        "runs": len(args.runs), "seeds": sorted(map(int, raw.seed.unique())),
        "heldout_gain_threshold_pct": 10.0, "threshold_passed": bool(heldout_ok),
        "selection_uses_confirmation_results": False,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
