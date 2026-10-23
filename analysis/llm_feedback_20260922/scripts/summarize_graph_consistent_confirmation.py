#!/usr/bin/env python3
"""Summarize fresh-pool confirmation of graph-consistent output selection."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


METRICS = {
    "shape": ("shape_error_percent", "min"),
    "trend": ("trend_error_percent", "min"),
    "alpha": ("alpha_precision_percent", "max"),
    "beta": ("beta_recall_percent", "max"),
    "auc": ("mle_xgboost_auc", "max"),
}


def provenance(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def metric(row: pd.Series, name: str) -> float:
    return float(row[METRICS[name][0]])


def gains(candidate: pd.Series, reference: pd.Series) -> dict[str, float]:
    result = {
        "shape": metric(reference, "shape") - metric(candidate, "shape"),
        "trend": metric(reference, "trend") - metric(candidate, "trend"),
        "alpha": metric(candidate, "alpha") - metric(reference, "alpha"),
        "beta": metric(candidate, "beta") - metric(reference, "beta"),
        "auc": metric(candidate, "auc") - metric(reference, "auc"),
    }
    dcr = "dcr_percent_50_ideal"
    result["dcr"] = abs(float(reference[dcr]) - 50) - abs(float(candidate[dcr]) - 50)
    return result


def parse_baselines(values: list[str]) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {}
    for value in values:
        dataset, filename = value.split("=", 1)
        parts.setdefault(dataset, []).append(pd.read_csv(filename))
    return {key: pd.concat(frames, ignore_index=True) for key, frames in parts.items()}


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
        root = Path(filename).parent.parent
        prov = provenance(root / "provenance.txt")
        frame = pd.read_csv(filename).set_index("method")
        graph = pd.read_csv(root / "graph_audit" / "graph_metrics.csv").set_index("method")
        candidate = frame.loc["graph"]
        entry: dict = {
            "dataset": prov["dataset"], "base": prov["base"],
            "old_selector": prov["old_selector"], "method": prov["method"],
            "seed": int(prov["seed"]), "root": str(root),
        }
        for name in METRICS:
            entry[name] = metric(candidate, name)
        for prefix in ("g", "r", "old"):
            for name, gain in gains(candidate, frame.loc[prefix.upper() if prefix in ("g", "r") else "old"]).items():
                entry[f"{prefix}_{name}_gain"] = gain
        for name in ("graph_relation_l1", "graph_coverage_gap", "graph_hard_excess"):
            entry[name] = float(graph.loc["graph", name])
            entry[f"{name}_gain_vs_old"] = float(graph.loc["old", name] - graph.loc["graph", name])
        records.append(entry)
    raw = pd.DataFrame(records)
    if raw.empty:
        raise RuntimeError("no completed fresh confirmation results")
    raw.to_csv(args.output / "per_seed.csv", index=False)

    summaries: list[dict] = []
    for dataset, frame in raw.groupby("dataset"):
        summary: dict = {
            "dataset": dataset, "base": frame.base.iloc[0],
            "old_selector": frame.old_selector.iloc[0], "method": frame.method.iloc[0],
            "seeds": len(frame),
        }
        numeric = [column for column in frame.columns if column not in {
            "dataset", "base", "old_selector", "method", "root", "seed"}]
        for column in numeric:
            summary[column] = float(frame[column].mean())
        for prefix in ("g", "r", "old"):
            for name in ("shape", "trend", "alpha", "beta", "auc", "dcr"):
                column = f"{prefix}_{name}_gain"
                summary[f"{column}_wins"] = int((frame[column] > 0).sum())
        for column in ("graph_relation_l1_gain_vs_old", "graph_coverage_gap_gain_vs_old",
                       "graph_hard_excess_gain_vs_old"):
            summary[f"{column}_wins"] = int((frame[column] > 1e-12).sum())
        if dataset in baselines:
            official = baselines[dataset]
            wins = 0
            for name, (column, direction) in METRICS.items():
                values = pd.to_numeric(official[column], errors="coerce")
                record = float(values.min() if direction == "min" else values.max())
                wins += summary[name] < record if direction == "min" else summary[name] > record
            summary["sota_metric_wins"] = int(wins)
        else:
            summary["sota_metric_wins"] = -1
        summary["dynamic_primary_confirmed"] = bool(
            summary["g_shape_gain"] > 0 and summary["g_trend_gain"] > 0
            and summary["r_shape_gain"] > 0 and summary["r_trend_gain"] > 0
            and summary["g_shape_gain_wins"] >= 2 and summary["g_trend_gain_wins"] >= 2)
        summary["graph_confirmed"] = bool(
            summary["graph_relation_l1_gain_vs_old"] > 0
            and summary["graph_relation_l1_gain_vs_old_wins"] >= 2
            and summary["graph_coverage_gap_gain_vs_old"] >= -1e-8
            and summary["graph_hard_excess_gain_vs_old"] >= -1e-8)
        summary["old_quality_preserved"] = bool(
            summary["old_shape_gain"] >= -0.03 and summary["old_trend_gain"] >= -0.10
            and summary["old_alpha_gain"] >= -0.50 and summary["old_beta_gain"] >= -0.50
            and summary["old_auc_gain"] >= -0.005 and summary["old_dcr_gain"] >= -2.0)
        summary["strict_combined_win"] = bool(
            summary["old_shape_gain"] > 0 and summary["old_trend_gain"] > 0
            and summary["old_shape_gain_wins"] >= 2 and summary["old_trend_gain_wins"] >= 2)
        summary["decision"] = (
            "comprehensive" if summary["graph_confirmed"] and summary["strict_combined_win"]
            and summary["dynamic_primary_confirmed"] else
            "coherent" if summary["graph_confirmed"] and summary["old_quality_preserved"]
            and summary["dynamic_primary_confirmed"] else "reject")
        summaries.append(summary)
    result = pd.DataFrame(summaries).sort_values("dataset")
    result.to_csv(args.output / "summary.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(
        result.to_dict(orient="records"), indent=2, default=str) + "\n")

    lines = [
        "# Fresh confirmation: graph-consistent output selection",
        "",
        "| dataset | dynamic + old + graph | Shape | Trend | Alpha | Beta | AUC | Shape/Trend gain vs G | gain vs old | graph relation/coverage gain vs old | graph wins | local SOTA wins | decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in result.iterrows():
        lines.append(
            f"| {row.dataset} | {row.base}+{row.old_selector}+{row.method} | "
            f"{row['shape']:.6f} | {row['trend']:.6f} | {row['alpha']:.6f} | "
            f"{row['beta']:.6f} | {row['auc']:.6f} | "
            f"{row.g_shape_gain:+.6f}/{row.g_trend_gain:+.6f} | "
            f"{row.old_shape_gain:+.6f}/{row.old_trend_gain:+.6f} | "
            f"{row.graph_relation_l1_gain_vs_old:+.6f}/{row.graph_coverage_gap_gain_vs_old:+.6f} | "
            f"{int(row.graph_relation_l1_gain_vs_old_wins)}/3 | {int(row.sota_metric_wins)}/5 | {row.decision} |")
    lines.extend([
        "",
        "`comprehensive` requires fresh-seed graph improvement plus paired Shape and Trend improvement over frozen C3. `coherent` allows a predeclared small C3 tolerance while retaining the dynamic-generator gain.",
    ])
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
