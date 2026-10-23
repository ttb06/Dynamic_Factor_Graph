#!/usr/bin/env python3
"""Fail-closed selector for the paired News factorized RFF-MMD screen."""
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


VARIANTS = (
    "mmd_marginal_one",
    "mmd_pair4_one",
    "mmd_pair8_one",
    "mmd_pair8_two",
)
EXPECTED_SEED = "20261350"
EXPECTED_SAMPLES = 8192
EXPECTED_STEPS = 50
ALIASES = {
    "shape": ("shape", "shape_score_percent", "column_shape_score_percent"),
    "trend": ("trend", "trend_score_percent", "column_pair_trend_score_percent"),
    "alpha": ("alpha", "alpha_precision_percent", "alpha_score_percent"),
    "beta": ("beta", "beta_recall_percent", "beta_score_percent"),
    "r2": ("r2", "mle_xgboost_r2", "xgboost_regression_r2", "utility_r2"),
    "nrmse": ("nrmse", "mle_xgboost_nrmse", "xgboost_regression_nrmse", "utility_nrmse"),
}


def _normal(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _column(fieldnames: list[str], aliases: tuple[str, ...]) -> str:
    indexed = {_normal(name): name for name in fieldnames}
    matches = list(dict.fromkeys(
        indexed[_normal(alias)] for alias in aliases if _normal(alias) in indexed
    ))
    if len(matches) != 1:
        raise ValueError(f"cannot resolve one of {aliases!r} from {fieldnames!r}")
    return matches[0]


def _number(value: object, context: str) -> float:
    try:
        number = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: non-numeric {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: non-finite {value!r}")
    return number


def _env(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            output[key.strip()] = value.strip()
    return output


def _paired_hash(path: Path) -> str:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"{path}: expected two hash lines")
    digests = [line.split()[0].casefold() for line in lines]
    if any(len(value) != 64 for value in digests) or len(set(digests)) != 1:
        raise ValueError(f"{path}: invalid or unequal paired hashes")
    return digests[0]


def _metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    method_column = _column(fieldnames, ("method",))
    columns = {key: _column(fieldnames, aliases) for key, aliases in ALIASES.items()}
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        method = str(row[method_column]).strip()
        if method in output:
            raise ValueError(f"{path}: duplicate method {method}")
        output[method] = {
            key: _number(row[column], f"{path}:{method}:{column}")
            for key, column in columns.items()
        }
    return output


def _mechanism(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row.get("method") == "dense_factor"]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one dense_factor row")
    output: dict[str, float] = {}
    for key, value in matches[0].items():
        if key.startswith("factor_mmd_") and value not in (None, ""):
            output[key] = _number(value, f"{path}:{key}")
    return output


def _oriented_delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    output = {key: float(left[key]) - float(right[key]) for key in ALIASES}
    output["nrmse"] *= -1.0
    return output


def _load(pattern: str) -> dict[str, dict[str, Any]]:
    paths = sorted(Path(path).resolve() for path in glob.glob(pattern) if Path(path).is_dir())
    runs: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            provenance = _env(path / "provenance.txt")
            variant = provenance.get("variant", "")
            if variant not in VARIANTS or variant in runs:
                raise ValueError(f"invalid/duplicate variant {variant!r}")
            if provenance.get("dataset") != "news":
                raise ValueError("dataset is not news")
            if provenance.get("seed") != EXPECTED_SEED:
                raise ValueError("unexpected seed")
            if provenance.get("samples") != str(EXPECTED_SAMPLES):
                raise ValueError("unexpected sample count")
            if provenance.get("steps") != str(EXPECTED_STEPS):
                raise ValueError("unexpected step count")
            evaluation = _metrics(path / "eval" / "metrics.csv")
            if set(evaluation) != {"B", "G", variant}:
                raise ValueError(f"unexpected evaluator methods {sorted(evaluation)}")
            metadata = json.loads((path / "candidate" / "run_metadata.json").read_text())
            reference_metadata = json.loads((path / "reference" / "run_metadata.json").read_text())
            if int(metadata.get("samples", -1)) != EXPECTED_SAMPLES or int(metadata.get("steps", -1)) != EXPECTED_STEPS:
                raise ValueError("candidate metadata design mismatch")
            runtime = _number(metadata.get("runtime_seconds"), "candidate runtime")
            reference_runtime = _number(reference_metadata.get("runtime_seconds"), "reference runtime")
            runs[variant] = {
                "path": str(path),
                "hash": _paired_hash(path / "unconditional_sha256.txt"),
                "evaluation": evaluation,
                "metadata": metadata,
                "mechanism": _mechanism(path / "candidate" / "mechanism_metrics.csv"),
                "runtime_ratio": runtime / reference_runtime,
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    missing = sorted(set(VARIANTS) - set(runs))
    if len(paths) != len(VARIANTS) or missing:
        errors.append(f"matched={len(paths)}, missing={missing}")
    if errors:
        raise ValueError("invalid FRMMD round:\n- " + "\n- ".join(errors))
    return runs


def _candidate(variant: str, run: Mapping[str, Any]) -> dict[str, Any]:
    values = run["evaluation"]
    versus_g = _oriented_delta(values[variant], values["G"])
    versus_b = _oriented_delta(values[variant], values["B"])
    mechanism = run["mechanism"]
    metadata = run["metadata"]
    reasons: list[str] = []
    expected_calls = 2 if variant.endswith("_two") else 1
    expected_pairs = 0 if variant == "mmd_marginal_one" else int(variant.split("pair", 1)[1].split("_", 1)[0])
    operator = metadata.get("factor_mmd")
    if not isinstance(operator, dict):
        reasons.append("missing factor_mmd metadata")
        operator = {}
    if operator.get("heldout_used_for_fitting") is not False:
        reasons.append("heldout_used_for_fitting is not false")
    if operator.get("target_read_only") is not True:
        reasons.append("target_read_only is not true")
    heads = operator.get("heads", []) if isinstance(operator.get("heads", []), list) else []
    pair_count = sum(isinstance(head, dict) and head.get("kind") == "pair" for head in heads)
    if expected_pairs == 0 and pair_count:
        reasons.append(f"marginal arm retained {pair_count} pair heads")
    if expected_pairs > 0 and not (1 <= pair_count <= expected_pairs):
        reasons.append(f"pair head count {pair_count} outside [1,{expected_pairs}]")
    if mechanism.get("factor_mmd_calls") != expected_calls:
        reasons.append(f"factor_mmd_calls={mechanism.get('factor_mmd_calls')}, expected {expected_calls}")
    if mechanism.get("factor_mmd_target_preserved") != 1:
        reasons.append("target was not preserved")
    if mechanism.get("factor_mmd_selected_cells", 0.0) <= 0:
        reasons.append("no selected cells")
    if mechanism.get("factor_mmd_predicted_energy_decrease", 0.0) <= 0:
        reasons.append("non-positive predicted MMD decrease")
    if run["runtime_ratio"] > 2.5:
        reasons.append(f"runtime ratio {run['runtime_ratio']:.3f} exceeds 2.5x G")
    for comparator, delta in (("G", versus_g), ("B", versus_b)):
        if delta["shape"] <= 0:
            reasons.append(f"Shape vs {comparator} is not positive ({delta['shape']:.6f})")
        if delta["trend"] <= 0:
            reasons.append(f"Trend vs {comparator} is not positive ({delta['trend']:.6f})")
        floors = {"alpha": -0.05, "beta": -0.10, "r2": -0.10, "nrmse": -0.05}
        for metric, floor in floors.items():
            if delta[metric] < floor:
                reasons.append(f"{metric} vs {comparator} {delta[metric]:.6f} < {floor}")
    score = (
        0.35 * versus_g["shape"] + 0.35 * versus_g["trend"]
        + 0.10 * versus_g["alpha"] + 0.05 * versus_g["beta"]
        + 0.10 * versus_g["r2"] + 0.05 * versus_g["nrmse"]
    )
    return {
        "variant": variant,
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "score": score,
        "versus_g_oriented": versus_g,
        "versus_b_oriented": versus_b,
        "runtime_ratio": run["runtime_ratio"],
        "paired_hash": run["hash"],
        "mechanism": mechanism,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-winners", type=int, default=2)
    args = parser.parse_args(argv)
    if not (0 <= args.max_winners <= 2):
        raise ValueError("max winners must be in [0,2]")
    runs = _load(args.input_glob)
    candidates = [_candidate(variant, runs[variant]) for variant in VARIANTS]
    candidates.sort(key=lambda item: (not item["eligible"], -item["score"], item["variant"]))
    winners: list[str] = []
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        candidate["selected"] = candidate["eligible"] and len(winners) < args.max_winners
        if candidate["selected"]:
            winners.append(candidate["variant"])
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True,
        "seed": EXPECTED_SEED,
        "samples": EXPECTED_SAMPLES,
        "steps": EXPECTED_STEPS,
        "winners": winners,
        "candidates": candidates,
        "input_directories": [runs[variant]["path"] for variant in VARIANTS],
    }
    (args.output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# News factorized RFF-MMD selection", "",
        f"- Selected: **{', '.join(winners) if winners else 'none'}**", "",
        "| Rank | Selected | Arm | Eligible | Score | dShape/G | dTrend/G | dAlpha/G | dBeta/G | dR2/G | dNRMSE+/G | Runtime x |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in candidates:
        delta = candidate["versus_g_oriented"]
        lines.append(
            "| {rank} | {selected} | {variant} | {eligible} | {score:.6f} | {shape:.6f} | {trend:.6f} | {alpha:.6f} | {beta:.6f} | {r2:.6f} | {nrmse:.6f} | {runtime:.3f} |".format(
                rank=candidate["rank"], selected="yes" if candidate["selected"] else "",
                variant=candidate["variant"], eligible="yes" if candidate["eligible"] else "no",
                score=candidate["score"], shape=delta["shape"], trend=delta["trend"],
                alpha=delta["alpha"], beta=delta["beta"], r2=delta["r2"],
                nrmse=delta["nrmse"], runtime=candidate["runtime_ratio"],
            )
        )
    lines.extend(["", "## Rejections", ""])
    for candidate in candidates:
        if not candidate["eligible"]:
            lines.append(f"- **{candidate['variant']}**: " + "; ".join(candidate["rejection_reasons"]))
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
