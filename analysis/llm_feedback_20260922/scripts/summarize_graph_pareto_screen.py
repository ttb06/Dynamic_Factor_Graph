#!/usr/bin/env python3
"""Summarize graph-Pareto anchor screens and select one variant per dataset."""
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
METHODS = ("pa_g0_q10", "pa_g10_q15", "pa_g25_q20", "pa_g50_q30")


def provenance(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1); result[key] = value
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
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--title", default="Graph-Pareto anchored screen")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    methods = tuple(value for value in args.methods.split(",") if value)
    if not methods:
        raise ValueError("at least one method is required")
    args.output.mkdir(parents=True, exist_ok=False)
    baselines = parse_baselines(args.baseline)
    raw: list[dict] = []
    for filename in sorted(glob.glob(args.input_glob)):
        root = Path(filename).parent.parent
        prov = provenance(root / "provenance.txt")
        metrics = pd.read_csv(filename).set_index("method")
        graph = pd.read_csv(root / "graph_audit" / "graph_metrics.csv").set_index("method")
        for method in methods:
            candidate = metrics.loc[method]
            metadata = json.loads((root / method / "selection_metadata.json").read_text())
            best = metadata["selection"]["best"]
            anchor = metadata["selection"]["anchor"]
            entry = {
                "dataset": prov["dataset"], "base": prov["base"],
                "old_selector": prov["old_selector"], "method": method,
                "seed": int(prov["seed"]), "root": str(root),
                "train_feasible": bool(best["feasible"]),
                "activation_skipped": bool(metadata["selection"].get(
                    "activation_skipped", False)),
                "feasible_draws": int(metadata["selection"]["feasible_draws"]),
                "train_quality_gain": float(anchor["quality"] - best["quality"]),
            }
            for name in METRICS:
                entry[name] = metric(candidate, name)
            for prefix, reference in (("g", metrics.loc["G"]), ("r", metrics.loc["R"]),
                                      ("old", metrics.loc["old"])):
                for name, value in gains(candidate, reference).items():
                    entry[f"{prefix}_{name}_gain"] = value
            for column in ("graph_relation_l1", "graph_coverage_gap", "graph_hard_excess"):
                entry[column] = float(graph.loc[method, column])
                entry[f"{column}_gain_vs_old"] = float(graph.loc["old", column] - graph.loc[method, column])
            raw.append(entry)
    frame = pd.DataFrame(raw)
    if frame.empty:
        raise RuntimeError("no graph-Pareto results")
    frame.to_csv(args.output / "per_seed.csv", index=False)

    summaries = []
    for (dataset, method), group in frame.groupby(["dataset", "method"]):
        summary = {"dataset": dataset, "base": group.base.iloc[0],
                   "old_selector": group.old_selector.iloc[0], "method": method,
                   "seeds": len(group), "feasible_seeds": int(group.train_feasible.sum()),
                   "activation_skipped_seeds": int(group.activation_skipped.sum()),
                   "min_feasible_draws": int(group.feasible_draws.min())}
        numeric = [column for column in group.columns if column not in {
            "dataset", "base", "old_selector", "method", "root", "seed",
            "train_feasible", "activation_skipped", "feasible_draws"}]
        for column in numeric:
            summary[column] = float(group[column].mean())
        for prefix in ("g", "r", "old"):
            for name in ("shape", "trend", "alpha", "beta", "auc", "dcr"):
                column = f"{prefix}_{name}_gain"
                summary[f"{column}_wins"] = int((group[column] > 0).sum())
        for column in ("graph_relation_l1_gain_vs_old", "graph_coverage_gap_gain_vs_old",
                       "graph_hard_excess_gain_vs_old"):
            summary[f"{column}_wins"] = int((group[column] > 1e-12).sum())
        if dataset in baselines:
            official = baselines[dataset]; wins = 0
            for name, (column, direction) in METRICS.items():
                values = pd.to_numeric(official[column], errors="coerce")
                record = float(values.min() if direction == "min" else values.max())
                wins += summary[name] < record if direction == "min" else summary[name] > record
            summary["sota_metric_wins"] = int(wins)
        else:
            summary["sota_metric_wins"] = -1
        summary["constraint_pass"] = bool(
            summary["feasible_seeds"] == summary["seeds"]
            and summary["graph_relation_l1_gain_vs_old"] >= -1e-12
            and (summary["graph_relation_l1_gain_vs_old_wins"]
                 + summary["activation_skipped_seeds"]) >= 2
            and summary["graph_coverage_gap_gain_vs_old"] >= -1e-8
            and summary["graph_hard_excess_gain_vs_old"] >= -1e-8)
        summary["dynamic_pass"] = bool(
            summary["g_shape_gain"] > 0 and summary["g_trend_gain"] > 0
            and summary["g_shape_gain_wins"] >= 2 and summary["g_trend_gain_wins"] >= 2)
        summary["old_floor"] = bool(
            summary["old_shape_gain"] >= -0.03 and summary["old_trend_gain"] >= -0.10
            and summary["old_alpha_gain"] >= -0.50 and summary["old_beta_gain"] >= -0.50
            and summary["old_auc_gain"] >= -0.005 and summary["old_dcr_gain"] >= -2.0)
        summary["old_primary_win"] = bool(
            summary["old_shape_gain"] > 0 and summary["old_trend_gain"] > 0
            and summary["old_shape_gain_wins"] >= 2 and summary["old_trend_gain_wins"] >= 2)
        summary["score"] = float(
            100 * summary["old_shape_gain"] / max(summary["shape"], 1e-9)
            + 100 * summary["old_trend_gain"] / max(summary["trend"], 1e-9)
            + .25 * summary["old_alpha_gain"] + .25 * summary["old_beta_gain"]
            + 20 * summary["old_auc_gain"] + .05 * summary["old_dcr_gain"]
            + 25 * summary["graph_relation_l1_gain_vs_old"])
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries).sort_values(["dataset", "method"])
    summary_frame.to_csv(args.output / "summary.csv", index=False)
    winners = []
    for _, group in summary_frame.groupby("dataset"):
        ranked = group.sort_values(
            ["constraint_pass", "old_floor", "old_primary_win", "dynamic_pass",
             "sota_metric_wins", "score"],
            ascending=[False, False, False, False, False, False], kind="stable")
        winners.append(ranked.iloc[0])
    winner_frame = pd.DataFrame(winners).sort_values("dataset")
    winner_frame.to_csv(args.output / "winners.csv", index=False)
    with (args.output / "winners.tsv").open("w") as handle:
        handle.write("dataset\tbase\told_selector\tmethod\tconstraint_pass\told_floor\told_primary_win\tdynamic_pass\tscore\n")
        for _, row in winner_frame.iterrows():
            handle.write(f"{row.dataset}\t{row.base}\t{row.old_selector}\t{row.method}\t"
                         f"{str(bool(row.constraint_pass)).lower()}\t{str(bool(row.old_floor)).lower()}\t"
                         f"{str(bool(row.old_primary_win)).lower()}\t{str(bool(row.dynamic_pass)).lower()}\t{row.score:.9f}\n")
    lines = [
        f"# {args.title}", "",
        "| dataset | variant | feasible seeds | Shape/Trend gain vs old | graph relation gain | Alpha/Beta/AUC gain vs old | SOTA wins | gates |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in winner_frame.iterrows():
        lines.append(
            f"| {row.dataset} | {row.method} | {int(row.feasible_seeds)}/3 | "
            f"{row.old_shape_gain:+.6f}/{row.old_trend_gain:+.6f} | "
            f"{row.graph_relation_l1_gain_vs_old:+.6f} | "
            f"{row.old_alpha_gain:+.6f}/{row.old_beta_gain:+.6f}/{row.old_auc_gain:+.6f} | "
            f"{int(row.sota_metric_wins)}/5 | constraint={bool(row.constraint_pass)}, "
            f"floor={bool(row.old_floor)}, primary={bool(row.old_primary_win)}, dynamic={bool(row.dynamic_pass)} |")
    lines.extend(["", "Selection is fail-closed on all train-only anchor constraints before metric ranking."])
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.output / "summary.json").write_text(json.dumps(
        summary_frame.to_dict(orient="records"), indent=2, default=str) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
