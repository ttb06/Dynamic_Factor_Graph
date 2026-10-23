#!/usr/bin/env python3
"""Fail-closed selector for paired cross-dataset action-space consensus."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASETS = ("news", "default")
VARIANTS = ("aligned15", "aligned30", "pcgrad15", "pcgrad30")
SEEDS = {"news": "20261360", "default": "20261361"}
SAMPLES = 8192
STEPS = 50
ALIASES = {
    "news": {
        "shape": ("shape", "shape_score_percent", "column_shape_score_percent"),
        "trend": ("trend", "trend_score_percent", "column_pair_trend_score_percent"),
        "alpha": ("alpha", "alpha_precision_percent", "alpha_score_percent"),
        "beta": ("beta", "beta_recall_percent", "beta_score_percent"),
        "r2": ("r2", "xgboost_regression_r2", "mle_xgboost_r2"),
        "nrmse": ("nrmse", "xgboost_regression_nrmse", "mle_xgboost_nrmse"),
    },
    "default": {
        "shape": ("shape_error_percent",),
        "trend": ("trend_error_percent",),
        "alpha": ("alpha_precision_percent",),
        "beta": ("beta_recall_percent",),
        "auc": ("mle_xgboost_auc", "auc"),
    },
}


def normal(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def resolve(fieldnames: list[str], aliases: tuple[str, ...]) -> str:
    indexed = {normal(name): name for name in fieldnames}
    matches = list(dict.fromkeys(
        indexed[normal(alias)] for alias in aliases if normal(alias) in indexed
    ))
    if len(matches) != 1:
        raise ValueError(f"cannot resolve {aliases!r} from {fieldnames!r}")
    return matches[0]


def number(value: object, context: str) -> float:
    try:
        output = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: non-numeric {value!r}") from exc
    if not math.isfinite(output):
        raise ValueError(f"{context}: non-finite {value!r}")
    return output


def env(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            output[key.strip()] = value.strip()
    return output


def paired_hash(path: Path) -> str:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"{path}: expected two hashes")
    digests = [line.split()[0].casefold() for line in lines]
    if any(len(digest) != 64 for digest in digests) or len(set(digests)) != 1:
        raise ValueError(f"{path}: paired hashes differ")
    return digests[0]


def metrics(path: Path, dataset: str) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    method_column = resolve(fieldnames, ("method",))
    columns = {key: resolve(fieldnames, aliases) for key, aliases in ALIASES[dataset].items()}
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        method = str(row[method_column]).strip()
        if not method or method in output:
            raise ValueError(f"{path}: blank/duplicate method {method!r}")
        output[method] = {
            key: number(row[column], f"{path}:{method}:{column}")
            for key, column in columns.items()
        }
    return output


def mechanism(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("method") == "dense_factor"]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one dense_factor row")
    output: dict[str, float] = {}
    for key, value in matches[0].items():
        if key.startswith("consensus_") and value not in (None, ""):
            output[key] = number(value, f"{path}:{key}")
    return output


def oriented(dataset: str, left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    output = {key: float(left[key]) - float(right[key]) for key in ALIASES[dataset]}
    if dataset == "news":
        output["nrmse"] *= -1.0
    else:
        output["shape"] *= -1.0
        output["trend"] *= -1.0
    return output


def load(pattern: str) -> dict[tuple[str, str], dict[str, Any]]:
    paths = sorted(Path(value).resolve() for value in glob.glob(pattern) if Path(value).is_dir())
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            provenance = env(path / "provenance.txt")
            dataset = provenance.get("dataset", "")
            variant = provenance.get("variant", "")
            key = (dataset, variant)
            if dataset not in DATASETS or variant not in VARIANTS or key in runs:
                raise ValueError(f"invalid/duplicate {key}")
            if provenance.get("seed") != SEEDS[dataset]:
                raise ValueError("unexpected seed")
            if provenance.get("samples") != str(SAMPLES) or provenance.get("steps") != str(STEPS):
                raise ValueError("design mismatch")
            eval_path = path / "eval" / ("metrics.csv" if dataset == "news" else "baseline_metrics.csv")
            evaluation = metrics(eval_path, dataset)
            if set(evaluation) != {"B", "G", variant}:
                raise ValueError(f"methods={sorted(evaluation)}")
            metadata = json.loads((path / "candidate" / "run_metadata.json").read_text())
            reference_metadata = json.loads((path / "reference" / "run_metadata.json").read_text())
            runtime = number(metadata.get("runtime_seconds"), "candidate runtime")
            reference_runtime = number(reference_metadata.get("runtime_seconds"), "reference runtime")
            runs[key] = {
                "path": str(path),
                "evaluation": evaluation,
                "metadata": metadata,
                "mechanism": mechanism(path / "candidate" / "mechanism_metrics.csv"),
                "runtime_ratio": runtime / reference_runtime,
                "hash": paired_hash(path / "unconditional_sha256.txt"),
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    expected = {(dataset, variant) for dataset in DATASETS for variant in VARIANTS}
    if len(paths) != len(expected) or set(runs) != expected:
        errors.append(f"matched={len(paths)}, missing={sorted(expected-set(runs))}")
    if errors:
        raise ValueError("invalid consensus round:\n- " + "\n- ".join(errors))
    return runs


def candidate(variant: str, runs: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    expected_mode = "aligned" if variant.startswith("aligned") else "pcgrad"
    expected_ratio = 0.15 if variant.endswith("15") else 0.30
    reasons: list[str] = []
    per_dataset: dict[str, Any] = {}
    shape_g: list[float] = []
    trend_g: list[float] = []
    alpha_g: list[float] = []
    runtime: list[float] = []
    for dataset in DATASETS:
        run = runs[(dataset, variant)]
        values = run["evaluation"]
        versus_g = oriented(dataset, values[variant], values["G"])
        versus_b = oriented(dataset, values[variant], values["B"])
        metadata = run["metadata"].get("consensus")
        mech = run["mechanism"]
        if not isinstance(metadata, dict):
            reasons.append(f"{dataset}: missing consensus metadata")
            metadata = {}
        if metadata.get("mode") != expected_mode:
            reasons.append(f"{dataset}: mode={metadata.get('mode')!r}")
        if abs(float(metadata.get("witness_ratio", -1)) - expected_ratio) > 1e-12:
            reasons.append(f"{dataset}: witness_ratio={metadata.get('witness_ratio')!r}")
        if metadata.get("heldout_used_for_fitting") is not False:
            reasons.append(f"{dataset}: heldout flag invalid")
        if mech.get("consensus_calls") != 1:
            reasons.append(f"{dataset}: consensus_calls={mech.get('consensus_calls')}")
        if mech.get("consensus_target_followed_reference") != 1:
            reasons.append(f"{dataset}: target diverged from reference")
        if mech.get("consensus_witness_selected_cells", 0) <= 0:
            reasons.append(f"{dataset}: witness selected no cells")
        if mech.get("consensus_extra_cell_events", 0) <= 0:
            reasons.append(f"{dataset}: fusion added no cells")
        if mech.get("consensus_extra_to_reference_norm_mean", 2.0) > expected_ratio + 1e-6:
            reasons.append(f"{dataset}: norm budget exceeded")
        if run["runtime_ratio"] > 2.5:
            reasons.append(f"{dataset}: runtime {run['runtime_ratio']:.3f}x > 2.5x")
        for comparator, delta in (("G", versus_g), ("B", versus_b)):
            if delta["shape"] <= 0:
                reasons.append(f"{dataset}: Shape vs {comparator} {delta['shape']:.6f} not positive")
            if delta["trend"] <= 0:
                reasons.append(f"{dataset}: Trend vs {comparator} {delta['trend']:.6f} not positive")
            floors = {"alpha": -0.10, "beta": -0.15}
            if dataset == "news":
                floors.update({"r2": -0.10, "nrmse": -0.05})
            else:
                floors.update({"auc": -0.005})
            for metric, floor in floors.items():
                if delta[metric] < floor:
                    reasons.append(f"{dataset}: {metric} vs {comparator} {delta[metric]:.6f} < {floor}")
        shape_g.append(versus_g["shape"])
        trend_g.append(versus_g["trend"])
        alpha_g.append(versus_g["alpha"])
        runtime.append(run["runtime_ratio"])
        per_dataset[dataset] = {
            "candidate_raw": values[variant],
            "versus_g_oriented": versus_g,
            "versus_b_oriented": versus_b,
            "runtime_ratio": run["runtime_ratio"],
            "paired_hash": run["hash"],
            "mechanism": mech,
        }
    score = 0.4 * statistics.fmean(shape_g) + 0.4 * statistics.fmean(trend_g) + 0.2 * statistics.fmean(alpha_g)
    return {
        "variant": variant,
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "score": score,
        "mean_shape_vs_g": statistics.fmean(shape_g),
        "mean_trend_vs_g": statistics.fmean(trend_g),
        "mean_alpha_vs_g": statistics.fmean(alpha_g),
        "mean_runtime_ratio": statistics.fmean(runtime),
        "per_dataset": per_dataset,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-winners", type=int, default=2)
    args = parser.parse_args(argv)
    runs = load(args.input_glob)
    candidates = [candidate(variant, runs) for variant in VARIANTS]
    candidates.sort(key=lambda value: (not value["eligible"], -value["score"], value["variant"]))
    winners: list[str] = []
    for rank, value in enumerate(candidates, start=1):
        value["rank"] = rank
        value["selected"] = value["eligible"] and len(winners) < args.max_winners
        if value["selected"]:
            winners.append(value["variant"])
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True,
        "datasets": list(DATASETS),
        "variants": list(VARIANTS),
        "seeds": SEEDS,
        "winners": winners,
        "candidates": candidates,
        "input_directories": [runs[(dataset, variant)]["path"] for dataset in DATASETS for variant in VARIANTS],
    }
    (args.output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Action-space consensus selection", "",
        f"- Selected: **{', '.join(winners) if winners else 'none'}**", "",
        "| Rank | Selected | Arm | Eligible | Score | mean dShape/G | mean dTrend/G | mean dAlpha/G | Runtime x |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for value in candidates:
        lines.append(
            f"| {value['rank']} | {'yes' if value['selected'] else ''} | {value['variant']} | {'yes' if value['eligible'] else 'no'} | {value['score']:.6f} | {value['mean_shape_vs_g']:.6f} | {value['mean_trend_vs_g']:.6f} | {value['mean_alpha_vs_g']:.6f} | {value['mean_runtime_ratio']:.3f} |"
        )
    lines.extend(["", "## Rejections", ""])
    for value in candidates:
        if not value["eligible"]:
            lines.append(f"- **{value['variant']}**: " + "; ".join(value["rejection_reasons"]))
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(args.output / "SUMMARY.md")
    print(f"Selected {len(winners)}: {', '.join(winners) or 'none'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
