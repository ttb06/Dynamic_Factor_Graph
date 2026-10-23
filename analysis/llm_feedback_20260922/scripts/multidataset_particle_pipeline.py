#!/usr/bin/env python3
"""Select and summarize baseline-rich cross-dataset particle experiments."""
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


def provenance(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def parse_baselines(values: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {}
    for value in values:
        dataset, filename = value.split("=", 1)
        frame = pd.read_csv(filename)
        frame["source_file"] = filename
        frames.setdefault(dataset, []).append(frame)
    return {dataset: pd.concat(parts, ignore_index=True) for dataset, parts in frames.items()}


def best_baseline(frame: pd.DataFrame) -> dict[str, float]:
    result = {}
    for name, (column, direction) in METRICS.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        result[name] = float(values.min() if direction == "min" else values.max())
    return result


def value(row: pd.Series, metric: str) -> float:
    return float(row[METRICS[metric][0]])


def gains(candidate: pd.Series, reference: pd.Series) -> dict[str, float]:
    result = {
        "shape_gain": value(reference, "shape") - value(candidate, "shape"),
        "trend_gain": value(reference, "trend") - value(candidate, "trend"),
        "alpha_gain": value(candidate, "alpha") - value(reference, "alpha"),
        "beta_gain": value(candidate, "beta") - value(reference, "beta"),
        "auc_gain": value(candidate, "auc") - value(reference, "auc"),
    }
    dcr_column = "dcr_percent_50_ideal"
    result["dcr_gain"] = (
        abs(float(reference[dcr_column]) - 50.0)
        - abs(float(candidate[dcr_column]) - 50.0)
    )
    return result


def screen_select(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    baselines = parse_baselines(args.baseline)
    records: list[dict] = []
    candidate_methods = {"gt4", "int20_t005", "marg4", "trend4", "int50_t0"}
    for filename in sorted(glob.glob(args.input_glob)):
        metrics_path = Path(filename)
        root = metrics_path.parent.parent
        prov = provenance(root / "provenance.txt")
        dataset, base = prov["dataset"], prov["base"]
        frame = pd.read_csv(metrics_path).set_index("method")
        reference = frame.loc["G"]
        official = best_baseline(baselines[dataset])
        for method in sorted(candidate_methods & set(frame.index)):
            candidate = frame.loc[method]
            delta = gains(candidate, reference)
            primary_pass = delta["shape_gain"] > 0 and delta["trend_gain"] > 0
            floor_pass = (
                delta["alpha_gain"] >= -0.75
                and delta["beta_gain"] >= -1.0
                and delta["auc_gain"] >= -0.01
                and delta["dcr_gain"] >= -5.0
            )
            sota_wins = sum([
                value(candidate, "shape") < official["shape"],
                value(candidate, "trend") < official["trend"],
                value(candidate, "alpha") > official["alpha"],
                value(candidate, "beta") > official["beta"],
                value(candidate, "auc") > official["auc"],
            ])
            score = (
                100.0 * delta["shape_gain"] / max(value(reference, "shape"), 1e-9)
                + 100.0 * delta["trend_gain"] / max(value(reference, "trend"), 1e-9)
                + 0.25 * delta["alpha_gain"] + 0.25 * delta["beta_gain"]
                + 20.0 * delta["auc_gain"] + 0.05 * delta["dcr_gain"]
            )
            records.append({
                "dataset": dataset,
                "base": base,
                "method": method,
                "screen_task": prov["array_task_id"],
                "root": str(root),
                **{metric: value(candidate, metric) for metric in METRICS},
                **delta,
                "primary_pass": bool(primary_pass),
                "floor_pass": bool(floor_pass),
                "sota_metric_wins": int(sota_wins),
                "score": float(score),
            })
    candidates = pd.DataFrame(records)
    if candidates.empty:
        raise RuntimeError("no completed screen candidates")
    candidates.to_csv(output / "candidates.csv", index=False)

    winners = []
    for dataset, frame in candidates.groupby("dataset"):
        ranked = frame.sort_values(
            ["primary_pass", "floor_pass", "sota_metric_wins", "score"],
            ascending=[False, False, False, False], kind="stable")
        winners.append(ranked.iloc[0])
    winner_frame = pd.DataFrame(winners).sort_values("dataset")
    winner_frame.to_csv(output / "winners.csv", index=False)
    with (output / "winners.tsv").open("w") as handle:
        handle.write("dataset\tbase\tmethod\tscreen_task\tprimary_pass\tfloor_pass\tscore\n")
        for _, row in winner_frame.iterrows():
            handle.write(
                f"{row.dataset}\t{row.base}\t{row.method}\t{row.screen_task}\t"
                f"{str(bool(row.primary_pass)).lower()}\t{str(bool(row.floor_pass)).lower()}\t{row.score:.9f}\n"
            )

    lines = [
        "# Multi-dataset particle screen selection",
        "",
        "| dataset | base | selector | Shape | Trend | Alpha | Beta | AUC | Shape gain vs G | Trend gain vs G | SOTA metric wins | primary/floor |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in winner_frame.iterrows():
        lines.append(
            f"| {row.dataset} | {row.base} | {row.method} | {row['shape']:.6f} | "
            f"{row['trend']:.6f} | {row['alpha']:.6f} | {row['beta']:.6f} | "
            f"{row['auc']:.6f} | {row.shape_gain:+.6f} | {row.trend_gain:+.6f} | "
            f"{int(row.sota_metric_wins)}/5 | {bool(row.primary_pass)}/{bool(row.floor_pass)} |"
        )
    lines.extend([
        "",
        "Selection is lexicographic: paired Shape+Trend improvement, utility floors, number of local official-baseline records beaten, then a predeclared normalized score.",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (output / "selection.json").write_text(json.dumps(
        winner_frame.to_dict(orient="records"), indent=2, default=str) + "\n")
    print("\n".join(lines))


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def confirmation_summary(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    baselines = parse_baselines(args.baseline)
    raw: list[dict] = []
    for filename in sorted(glob.glob(args.input_glob)):
        metrics_path = Path(filename)
        root = metrics_path.parent.parent
        prov = provenance(root / "provenance.txt")
        frame = pd.read_csv(metrics_path).set_index("method")
        candidate, g_ref, r_ref = frame.loc["winner"], frame.loc["G"], frame.loc["R"]
        entry = {
            "dataset": prov["dataset"], "base": prov["base"],
            "method": prov["selector"], "seed": int(prov["seed"]),
            **{metric: value(candidate, metric) for metric in METRICS},
        }
        for prefix, reference in (("g", g_ref), ("r", r_ref)):
            for key, metric_value in gains(candidate, reference).items():
                entry[f"{prefix}_{key}"] = metric_value
        raw.append(entry)
    raw_frame = pd.DataFrame(raw)
    if raw_frame.empty:
        raise RuntimeError("no completed confirmation results")
    raw_frame.to_csv(output / "per_seed.csv", index=False)

    summaries = []
    for dataset, frame in raw_frame.groupby("dataset"):
        official = best_baseline(baselines[dataset])
        summary = {
            "dataset": dataset,
            "base": frame["base"].iloc[0],
            "method": frame["method"].iloc[0],
            "seeds": len(frame),
        }
        for metric in METRICS:
            summary[metric] = float(frame[metric].mean())
        for prefix in ("g", "r"):
            for metric in ("shape", "trend", "alpha", "beta", "auc", "dcr"):
                column = f"{prefix}_{metric}_gain"
                summary[column] = float(frame[column].mean())
                summary[f"{column}_wins"] = int((frame[column] > 0).sum())
            for metric in ("shape", "trend"):
                values = frame[f"{prefix}_{metric}_gain"].to_numpy(float)
                low, high = bootstrap_interval(values, 20261021 + len(summaries))
                summary[f"{prefix}_{metric}_ci_low"] = low
                summary[f"{prefix}_{metric}_ci_high"] = high
        summary["sota_metric_wins"] = int(sum([
            summary["shape"] < official["shape"],
            summary["trend"] < official["trend"],
            summary["alpha"] > official["alpha"],
            summary["beta"] > official["beta"],
            summary["auc"] > official["auc"],
        ]))
        summary["primary_confirmed"] = bool(
            summary["g_shape_gain"] > 0 and summary["g_trend_gain"] > 0
            and summary["r_shape_gain"] > 0 and summary["r_trend_gain"] > 0
            and summary["g_shape_gain_wins"] >= 2 and summary["g_trend_gain_wins"] >= 2
            and summary["r_shape_gain_wins"] >= 2 and summary["r_trend_gain_wins"] >= 2
        )
        summary["comprehensive_confirmed"] = bool(
            summary["primary_confirmed"]
            and summary["g_alpha_gain"] >= -0.5
            and summary["g_beta_gain"] >= -0.5
            and summary["g_auc_gain"] >= -0.005
            and summary["g_dcr_gain"] >= -2.0
        )
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries).sort_values("dataset")
    summary_frame.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(
        summary_frame.to_dict(orient="records"), indent=2, default=str) + "\n")

    lines = [
        "# Multi-dataset particle confirmation",
        "",
        "| dataset | base + selector | Shape | Trend | Alpha | Beta | AUC | Shape/Trend gain vs G | wins vs G | Shape/Trend gain vs R | wins vs R | local SOTA wins | decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary_frame.iterrows():
        decision = "comprehensive" if row.comprehensive_confirmed else ("primary" if row.primary_confirmed else "reject")
        lines.append(
            f"| {row.dataset} | {row.base}+{row.method} | {row['shape']:.6f} | {row['trend']:.6f} | "
            f"{row['alpha']:.6f} | {row['beta']:.6f} | {row['auc']:.6f} | "
            f"{row.g_shape_gain:+.6f}/{row.g_trend_gain:+.6f} | "
            f"{int(row.g_shape_gain_wins)}/{int(row.g_trend_gain_wins)} | "
            f"{row.r_shape_gain:+.6f}/{row.r_trend_gain:+.6f} | "
            f"{int(row.r_shape_gain_wins)}/{int(row.r_trend_gain_wins)} | "
            f"{int(row.sota_metric_wins)}/5 | {decision} |"
        )
    lines.extend([
        "",
        "`local SOTA wins` counts metrics exceeding the best locally evaluated official-generator row under the same 4,096-row protocol; it is contextual unless the confirmation mean clears the relevant threshold.",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["select", "confirm"], required=True)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--baseline", action="append", default=[], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.stage == "select":
        screen_select(args)
    else:
        confirmation_summary(args)


if __name__ == "__main__":
    main()
