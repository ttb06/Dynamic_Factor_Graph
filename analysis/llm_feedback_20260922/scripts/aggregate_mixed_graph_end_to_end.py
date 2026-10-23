#!/usr/bin/env python3
"""Aggregate paired end-to-end graph training controls."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["macro_violation", "heldout_violation_error", "calibration_violation_error", "split_median_error",
           "numeric_ks", "categorical_tv", "action_events", "unique_rows_actuated", "runtime_seconds"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-label", default="point calibration")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    pattern = re.compile(r"mixed_graph_(?:confirm|splitmedian)_(pretrained|full_continue|graph_trained)_s(\d+)_")
    for run in args.runs:
        match = pattern.search(run.name)
        if not match:
            raise ValueError(f"cannot parse run label: {run}")
        checkpoint, seed = match.group(1), int(match.group(2))
        frame = pd.read_csv(run/"method_metrics.csv")
        edge_frame = pd.read_csv(run/"edge_metrics.csv")
        graph = json.loads((run/"evaluation_graph.json").read_text())
        robust_targets = {}
        for edge in graph:
            rates = [float(edge[partition]["violation_rate"])
                     for partition in ("discovery", "counterexample", "calibration")
                     if partition in edge and np.isfinite(float(edge[partition].get("violation_rate", np.nan)))]
            if rates:
                robust_targets[edge["candidate_id"]] = float(np.median(rates))
        for method, label in (("unconditional", "unconditional"),
                              ("xstate_reward_calibrated", "dynamic_middle"),
                              ("final_reward_calibrated", "final_repair")):
            record = frame.loc[frame.method.eq(method)].iloc[0]
            edge_rows = edge_frame.loc[edge_frame.method.eq(method)]
            robust_error = float(np.mean([
                abs(float(row.violation_rate)-robust_targets[str(row.candidate_id)])
                for row in edge_rows.itertuples(index=False)
                if str(row.candidate_id) in robust_targets
            ]))
            rows.append({"checkpoint": checkpoint, "seed": seed, "mechanism": label,
                         **{metric: (robust_error if metric == "split_median_error" else float(record[metric]))
                            for metric in METRICS}})
    raw = pd.DataFrame(rows).sort_values(["checkpoint", "seed", "mechanism"])
    raw.to_csv(args.output/"per_seed_metrics.csv", index=False)
    summary = raw.groupby(["checkpoint", "mechanism"])[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(column).strip("_") if isinstance(column, tuple) else column
                       for column in summary.columns]
    summary.to_csv(args.output/"summary_metrics.csv", index=False)

    dynamic = raw.loc[raw.mechanism.eq("dynamic_middle")].pivot(
        index="seed", columns="checkpoint", values="heldout_violation_error"
    )
    gains = pd.DataFrame({
        "seed": dynamic.index,
        "gain_vs_pretrained_pct": 100*(dynamic.pretrained-dynamic.graph_trained)/dynamic.pretrained,
        "gain_vs_continued_training_pct": 100*(dynamic.full_continue-dynamic.graph_trained)/dynamic.full_continue,
    }).reset_index(drop=True)
    graph_mechanisms = raw.loc[raw.checkpoint.eq("graph_trained")].pivot(
        index="seed", columns="mechanism", values="heldout_violation_error"
    )
    gains["gain_vs_same_checkpoint_final_repair_pct"] = (
        100*(graph_mechanisms.final_repair-graph_mechanisms.dynamic_middle)/graph_mechanisms.final_repair
    ).to_numpy()
    gains.to_csv(args.output/"paired_gains.csv", index=False)

    lines = ["# Mixed-type graph-calibrated TabDiff: paired confirmation", "",
             f"All checkpoints use the same graph, {args.selection_label}, dynamic middle-step policy and paired sampling seeds.", "",
             "| Checkpoint | Mechanism | Seeds | Held-out error | Point-cal. error | Split-median error | Macro violation | KS | TV | Actions | Runtime |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.checkpoint} | {row.mechanism} | {raw.seed.nunique()} | "
            f"{row.heldout_violation_error_mean:.4f} ± {row.heldout_violation_error_std:.4f} | "
            f"{row.calibration_violation_error_mean:.4f} ± {row.calibration_violation_error_std:.4f} | "
            f"{row.split_median_error_mean:.4f} ± {row.split_median_error_std:.4f} | "
            f"{row.macro_violation_mean:.4f} | {row.numeric_ks_mean:.4f} | {row.categorical_tv_mean:.4f} | "
            f"{row.action_events_mean:.1f} | {row.runtime_seconds_mean:.2f} |"
        )
    lines += ["", "| Comparison for graph-trained dynamic middle-step | Mean paired held-out gain | Seeds won |",
              "| --- | ---: | ---: |",
              f"| vs pretrained dynamic | {gains.gain_vs_pretrained_pct.mean():.1f}% | "
              f"{int((gains.gain_vs_pretrained_pct > 0).sum())}/{len(gains)} |",
              f"| vs continued-training dynamic | {gains.gain_vs_continued_training_pct.mean():.1f}% | "
              f"{int((gains.gain_vs_continued_training_pct > 0).sum())}/{len(gains)} |",
              f"| vs same-checkpoint final repair | {gains.gain_vs_same_checkpoint_final_repair_pct.mean():.1f}% | "
              f"{int((gains.gain_vs_same_checkpoint_final_repair_pct > 0).sum())}/{len(gains)} |"]
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    (args.output/"run_summary.json").write_text(json.dumps({
        "runs": len(args.runs), "seeds": sorted(map(int, raw.seed.unique())),
        "selection": f"{args.selection_label} selected on train evidence, not heldout",
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
