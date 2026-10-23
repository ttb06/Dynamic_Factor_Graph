#!/usr/bin/env python3
"""Summarize and preselect graph-consistent output-selection variants."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "shape": ("shape_error_percent", "min"),
    "trend": ("trend_error_percent", "min"),
    "alpha": ("alpha_precision_percent", "max"),
    "beta": ("beta_recall_percent", "max"),
    "auc": ("mle_xgboost_auc", "max"),
}
CANDIDATES = ("graph_only", "gc_lite", "gc_bal", "gc_strong", "gc_cov", "gc_hard")
COMBINED = tuple(method for method in CANDIDATES if method.startswith("gc_"))


def provenance(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def parse_baselines(values: list[str]) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {}
    for value in values:
        dataset, filename = value.split("=", 1)
        frame = pd.read_csv(filename)
        parts.setdefault(dataset, []).append(frame)
    return {key: pd.concat(frames, ignore_index=True) for key, frames in parts.items()}


def best_baseline(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, (column, direction) in METRICS.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        result[name] = float(values.min() if direction == "min" else values.max())
    return result


def value(row: pd.Series, metric: str) -> float:
    return float(row[METRICS[metric][0]])


def gains(candidate: pd.Series, reference: pd.Series) -> dict[str, float]:
    result = {
        "shape": value(reference, "shape") - value(candidate, "shape"),
        "trend": value(reference, "trend") - value(candidate, "trend"),
        "alpha": value(candidate, "alpha") - value(reference, "alpha"),
        "beta": value(candidate, "beta") - value(reference, "beta"),
        "auc": value(candidate, "auc") - value(reference, "auc"),
    }
    dcr = "dcr_percent_50_ideal"
    result["dcr"] = abs(float(reference[dcr]) - 50.0) - abs(float(candidate[dcr]) - 50.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baselines = parse_baselines(args.baseline)

    records: list[dict] = []
    for filename in sorted(glob.glob(args.input_glob)):
        metrics_path = Path(filename)
        root = metrics_path.parent.parent
        prov = provenance(root / "provenance.txt")
        metrics = pd.read_csv(metrics_path).set_index("method")
        graph = pd.read_csv(root / "graph_audit" / "graph_metrics.csv").set_index("method")
        missing = set(("G", "R", "old", *CANDIDATES)) - set(metrics.index)
        if missing:
            raise RuntimeError(f"missing methods in {metrics_path}: {sorted(missing)}")
        old_graph = graph.loc["old"]
        for method in CANDIDATES:
            candidate = metrics.loc[method]
            entry: dict = {
                "dataset": prov["dataset"],
                "base": prov["base"],
                "old_selector": prov["old_selector"],
                "method": method,
                "seed": int(prov["seed"]),
                "root": str(root),
            }
            for metric in METRICS:
                entry[metric] = value(candidate, metric)
            for prefix, reference in (("g", metrics.loc["G"]),
                                      ("r", metrics.loc["R"]),
                                      ("old", metrics.loc["old"])):
                for metric, gain in gains(candidate, reference).items():
                    entry[f"{prefix}_{metric}_gain"] = gain
            for column in ("graph_relation_l1", "graph_coverage_gap", "graph_hard_excess"):
                entry[column] = float(graph.loc[method, column])
                entry[f"{column}_gain_vs_old"] = float(old_graph[column] - graph.loc[method, column])
            records.append(entry)
    raw = pd.DataFrame(records)
    if raw.empty:
        raise RuntimeError("no completed graph-consistent screen results")
    raw.to_csv(args.output / "per_seed.csv", index=False)

    summaries: list[dict] = []
    for (dataset, method), frame in raw.groupby(["dataset", "method"], sort=True):
        summary: dict = {
            "dataset": dataset,
            "base": frame["base"].iloc[0],
            "old_selector": frame["old_selector"].iloc[0],
            "method": method,
            "seeds": len(frame),
        }
        average_columns = [
            column for column in frame.columns
            if column in METRICS or column.endswith("_gain") or column.endswith("_gain_vs_old")
            or column in ("graph_relation_l1", "graph_coverage_gap", "graph_hard_excess")
        ]
        for column in average_columns:
            summary[column] = float(frame[column].mean())
        for prefix in ("g", "r", "old"):
            for metric in ("shape", "trend", "alpha", "beta", "auc", "dcr"):
                column = f"{prefix}_{metric}_gain"
                summary[f"{column}_wins"] = int((frame[column] > 0).sum())
        for column in ("graph_relation_l1_gain_vs_old", "graph_coverage_gap_gain_vs_old",
                       "graph_hard_excess_gain_vs_old"):
            summary[f"{column}_wins"] = int((frame[column] > 1e-12).sum())
        if dataset in baselines:
            official = best_baseline(baselines[dataset])
            summary["sota_metric_wins"] = int(sum([
                summary["shape"] < official["shape"],
                summary["trend"] < official["trend"],
                summary["alpha"] > official["alpha"],
                summary["beta"] > official["beta"],
                summary["auc"] > official["auc"],
            ]))
        else:
            summary["sota_metric_wins"] = -1
        summary["dynamic_primary_pass"] = bool(
            summary["g_shape_gain"] > 0 and summary["g_trend_gain"] > 0
            and summary["g_shape_gain_wins"] >= 2 and summary["g_trend_gain_wins"] >= 2)
        summary["graph_pass"] = bool(
            summary["graph_relation_l1_gain_vs_old"] > 0
            and summary["graph_relation_l1_gain_vs_old_wins"] >= 2
            and summary["graph_coverage_gap_gain_vs_old"] >= -1e-8
            and summary["graph_hard_excess_gain_vs_old"] >= -1e-8)
        summary["old_quality_floor"] = bool(
            summary["old_shape_gain"] >= -0.03
            and summary["old_trend_gain"] >= -0.10
            and summary["old_alpha_gain"] >= -0.50
            and summary["old_beta_gain"] >= -0.50
            and summary["old_auc_gain"] >= -0.005
            and summary["old_dcr_gain"] >= -2.0)
        summary["old_primary_improved"] = bool(
            summary["old_shape_gain"] > 0 and summary["old_trend_gain"] > 0)
        summary["score"] = float(
            100 * summary["old_shape_gain"] / max(summary["shape"], 1e-9)
            + 100 * summary["old_trend_gain"] / max(summary["trend"], 1e-9)
            + 0.25 * summary["old_alpha_gain"] + 0.25 * summary["old_beta_gain"]
            + 20 * summary["old_auc_gain"] + 0.05 * summary["old_dcr_gain"]
            + 25 * summary["graph_relation_l1_gain_vs_old"]
            + 25 * summary["graph_coverage_gap_gain_vs_old"]
            + 25 * summary["graph_hard_excess_gain_vs_old"])
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries).sort_values(["dataset", "method"])
    summary_frame.to_csv(args.output / "summary.csv", index=False)

    winners: list[pd.Series] = []
    for dataset, frame in summary_frame[summary_frame["method"].isin(COMBINED)].groupby("dataset"):
        ranked = frame.sort_values(
            ["graph_pass", "old_quality_floor", "old_primary_improved",
             "dynamic_primary_pass", "sota_metric_wins", "score"],
            ascending=[False, False, False, False, False, False], kind="stable")
        winners.append(ranked.iloc[0])
    winner_frame = pd.DataFrame(winners).sort_values("dataset")
    winner_frame.to_csv(args.output / "winners.csv", index=False)
    with (args.output / "winners.tsv").open("w") as handle:
        handle.write("dataset\tbase\told_selector\tmethod\tgraph_pass\told_quality_floor\tdynamic_primary_pass\tscore\n")
        for _, row in winner_frame.iterrows():
            handle.write(
                f"{row.dataset}\t{row.base}\t{row.old_selector}\t{row.method}\t"
                f"{str(bool(row.graph_pass)).lower()}\t{str(bool(row.old_quality_floor)).lower()}\t"
                f"{str(bool(row.dynamic_primary_pass)).lower()}\t{row.score:.9f}\n")

    lines = [
        "# Graph-consistent output-selection screen",
        "",
        "The selector uses only factors actuated by the dynamic sampler. `old` remains the frozen generic C3 selector; `graph_only` and `gc_*` are explicit ablations.",
        "",
        "| dataset | chosen combined variant | Shape | Trend | Alpha | Beta | AUC | Shape/Trend gain vs old | graph relation/coverage gain vs old | graph wins | local SOTA wins | gates |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in winner_frame.iterrows():
        lines.append(
            f"| {row.dataset} | {row.method} | {row['shape']:.6f} | {row['trend']:.6f} | "
            f"{row['alpha']:.6f} | {row['beta']:.6f} | {row['auc']:.6f} | "
            f"{row.old_shape_gain:+.6f}/{row.old_trend_gain:+.6f} | "
            f"{row.graph_relation_l1_gain_vs_old:+.6f}/{row.graph_coverage_gap_gain_vs_old:+.6f} | "
            f"{int(row.graph_relation_l1_gain_vs_old_wins)}/3 | {int(row.sota_metric_wins)}/5 | "
            f"graph={bool(row.graph_pass)}, floor={bool(row.old_quality_floor)}, dynamic={bool(row.dynamic_primary_pass)} |")
    lines.extend([
        "",
        "Predeclared selection is lexicographic: graph improvement, utility floor versus frozen C3, joint Shape/Trend improvement versus C3, retained gain versus dynamic G, SOTA metric count, then the scalar score.",
        "",
        "## All ablations",
        "",
        "| dataset | method | Shape/Trend gain vs G | Shape/Trend gain vs old | graph relation gain | graph coverage gain | graph/floor/dynamic |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for _, row in summary_frame.iterrows():
        lines.append(
            f"| {row.dataset} | {row.method} | {row.g_shape_gain:+.6f}/{row.g_trend_gain:+.6f} | "
            f"{row.old_shape_gain:+.6f}/{row.old_trend_gain:+.6f} | "
            f"{row.graph_relation_l1_gain_vs_old:+.6f} | {row.graph_coverage_gap_gain_vs_old:+.6f} | "
            f"{bool(row.graph_pass)}/{bool(row.old_quality_floor)}/{bool(row.dynamic_primary_pass)} |")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "summary.json").write_text(json.dumps(
        summary_frame.to_dict(orient="records"), indent=2, default=str) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
