#!/usr/bin/env python3
"""Fail-closed selector for the two-dataset CFRW validation screen.

The expected design is exactly News and Default crossed with
``W0,W4,W8,W12`` at one pre-declared validation seed.  Each GPU task runs the
locked G reference and one witness candidate sequentially, verifies their
unconditional CSV hashes locally, and evaluates ``B/G/W*`` together.  Thus all
candidate deltas are file-local and do not assume byte-identical sampling
across different GPU nodes.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASETS = ("news", "default")
WITNESS_VARIANTS = ("W0", "W4", "W8", "W12")
EXPECTED_SEEDS = {"news": "20261340", "default": "20261341"}
EXPECTED_SAMPLES = 8192
EXPECTED_STEPS = 50
EPSILON = 1e-12


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool
    aliases: tuple[str, ...]


METRICS: dict[str, tuple[MetricSpec, ...]] = {
    "news": (
        MetricSpec("shape", "Shape", True, ("shape", "shape_score", "shape_score_percent", "column_shapes", "column_shape_score", "column_shape_score_percent")),
        MetricSpec("trend", "Trend", True, ("trend", "trend_score", "trend_score_percent", "column_pair_trends", "column_pair_trend_score", "column_pair_trend_score_percent")),
        MetricSpec("alpha", "Alpha", True, ("alpha", "alpha_precision", "alpha_precision_percent", "alpha_score", "alpha_score_percent")),
        MetricSpec("beta", "Beta", True, ("beta", "beta_recall", "beta_recall_percent", "beta_score", "beta_score_percent")),
        MetricSpec("r2", "R2", True, ("r2", "r_squared", "mle_r2", "mle_xgboost_r2", "xgboost_regression_r2", "tstr_r2", "utility_r2")),
        MetricSpec("nrmse", "NRMSE", False, ("nrmse", "mle_nrmse", "mle_xgboost_nrmse", "xgboost_regression_nrmse", "tstr_nrmse", "utility_nrmse")),
    ),
    "default": (
        MetricSpec("shape", "Shape error", False, ("shape_error_percent",)),
        MetricSpec("trend", "Trend error", False, ("trend_error_percent",)),
        MetricSpec("alpha", "Alpha", True, ("alpha_precision_percent",)),
        MetricSpec("beta", "Beta", True, ("beta_recall_percent",)),
        MetricSpec("auc", "AUC", True, ("mle_xgboost_auc", "auc", "utility_auc")),
    ),
}


def normalize(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def resolve_column(fieldnames: Sequence[str], aliases: Iterable[str]) -> str:
    indexed = {normalize(name): name for name in fieldnames}
    matches = list(dict.fromkeys(indexed[normalize(alias)] for alias in aliases if normalize(alias) in indexed))
    if len(matches) != 1:
        raise ValueError(f"cannot resolve exactly one of {tuple(aliases)!r} from {fieldnames!r}")
    return matches[0]


def finite_float(value: object, *, context: str) -> float:
    try:
        number = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected numeric value, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected finite value, got {value!r}")
    return number


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def read_paired_hash(path: Path) -> str:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"{path}: expected exactly two paired hash lines")
    digests = [line.split()[0].casefold() for line in lines]
    for digest in digests:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{path}: invalid SHA-256 digest {digest!r}")
    if len(set(digests)) != 1:
        raise ValueError(f"{path}: reference/candidate unconditional samples differ")
    return digests[0]


def read_evaluation(path: Path, dataset: str) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty evaluator CSV")
    method_column = resolve_column(fieldnames, ("method",))
    columns = {
        metric.key: resolve_column(fieldnames, metric.aliases)
        for metric in METRICS[dataset]
    }
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        method = str(row.get(method_column, "")).strip()
        if not method:
            raise ValueError(f"{path}: blank method")
        if method in output:
            raise ValueError(f"{path}: duplicate method {method!r}")
        output[method] = {
            key: finite_float(row[column], context=f"{path}:{method}:{column}")
            for key, column in columns.items()
        }
    return output


def oriented(dataset: str, values: Mapping[str, float]) -> dict[str, float]:
    return {
        metric.key: float(values[metric.key]) * (1.0 if metric.higher_is_better else -1.0)
        for metric in METRICS[dataset]
    }


def difference(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError(f"metric-key mismatch: {sorted(left)} versus {sorted(right)}")
    return {key: float(left[key]) - float(right[key]) for key in left}


def read_dense_mechanism(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    matches = [row for row in rows if str(row.get("method", "")).strip() == "dense_factor"]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected exactly one dense_factor row")
    result: dict[str, float] = {}
    for key, value in matches[0].items():
        if key.startswith("residual_witness_") and value not in (None, ""):
            # Audit manifests (retained head IDs/reliabilities) are serialized
            # as JSON-like strings in this CSV.  Numeric diagnostics remain in
            # this compact map; full manifests are validated from metadata.
            try:
                result[key] = finite_float(value, context=f"{path}:{key}")
            except ValueError:
                continue
    return result


def inspect_witness(metadata: Mapping[str, Any], mechanism: Mapping[str, float], variant: str) -> list[str]:
    reasons: list[str] = []
    expected_pairs = int(variant[1:])
    expected_mode = "marginal" if variant == "W0" else "marginal_pair"
    witness = metadata.get("residual_witness")
    if not isinstance(witness, dict):
        return ["run_metadata.json has no residual_witness object"]
    if witness.get("mode") != expected_mode:
        reasons.append(f"mode={witness.get('mode')!r}, expected {expected_mode!r}")
    if witness.get("target_read_only") is not True:
        reasons.append("target_read_only is not true")
    if witness.get("heldout_used_for_fitting") is not False:
        reasons.append("witness heldout_used_for_fitting is not false")
    if int(witness.get("top_factors_per_row", -1)) != 1:
        reasons.append(f"top_factors_per_row={witness.get('top_factors_per_row')!r}, expected 1")
    if int(witness.get("max_pair_heads", -1)) != expected_pairs:
        reasons.append(f"max_pair_heads={witness.get('max_pair_heads')!r}, expected {expected_pairs}")

    heads = witness.get("heads")
    if not isinstance(heads, list) or not heads:
        reasons.append("no retained witness heads")
        heads = []
    target = witness.get("target_column")
    destinations = [head.get("destination") for head in heads if isinstance(head, dict)]
    if target and target in destinations:
        reasons.append("read-only target appears as an intervention destination")
    kinds = [head.get("kind") for head in heads if isinstance(head, dict)]
    if kinds.count("marginal") < 1:
        reasons.append("no marginal head retained")
    if variant == "W0":
        if kinds.count("target") or kinds.count("pair"):
            reasons.append("W0 retained a conditional head")
    else:
        pair_count = kinds.count("pair")
        if not (1 <= pair_count <= expected_pairs):
            reasons.append(f"pair head count {pair_count} is outside [1,{expected_pairs}]")

    required = {
        "residual_witness_calls": 1.0,
        "residual_witness_target_preserved": 1.0,
    }
    for key, expected in required.items():
        value = mechanism.get(key)
        if value is None or abs(value - expected) > EPSILON:
            reasons.append(f"{key}={value!r}, expected {expected:g}")
    if mechanism.get("residual_witness_selected_cells", 0.0) <= 0:
        reasons.append("witness selected no cells")
    if mechanism.get("residual_witness_selected_rows", 0.0) <= 0:
        reasons.append("witness selected no rows")
    return reasons


def discover_run_dirs(patterns: Sequence[str]) -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match).resolve()
            if path.is_dir():
                found[str(path).casefold()] = path
    return sorted(found.values(), key=lambda path: str(path).casefold())


def load_runs(paths: Sequence[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            provenance_path = path / "provenance.txt"
            values = parse_env(provenance_path)
            dataset = values.get("dataset", "").casefold()
            variant = values.get("variant", "")
            if dataset not in DATASETS or variant not in WITNESS_VARIANTS:
                raise ValueError(f"unexpected dataset/variant {dataset!r}/{variant!r}")
            key = (dataset, variant)
            if key in runs:
                raise ValueError(f"duplicate run for {dataset}/{variant}: {runs[key]['path']} and {path}")
            if values.get("seed") != EXPECTED_SEEDS[dataset]:
                raise ValueError(f"seed={values.get('seed')!r}, expected {EXPECTED_SEEDS[dataset]}")
            if values.get("samples") != str(EXPECTED_SAMPLES):
                raise ValueError(f"samples={values.get('samples')!r}, expected {EXPECTED_SAMPLES}")
            if values.get("steps") != str(EXPECTED_STEPS):
                raise ValueError(f"steps={values.get('steps')!r}, expected {EXPECTED_STEPS}")

            reference_metadata_path = path / "reference" / "run_metadata.json"
            candidate_metadata_path = path / "candidate" / "run_metadata.json"
            mechanism_path = path / "candidate" / "mechanism_metrics.csv"
            evaluation_path = path / "eval" / ("metrics.csv" if dataset == "news" else "baseline_metrics.csv")
            reference_metadata = json.loads(reference_metadata_path.read_text())
            candidate_metadata = json.loads(candidate_metadata_path.read_text())
            reference_runtime = finite_float(reference_metadata.get("runtime_seconds"), context=f"{reference_metadata_path}:runtime_seconds")
            candidate_runtime = finite_float(candidate_metadata.get("runtime_seconds"), context=f"{candidate_metadata_path}:runtime_seconds")
            if reference_runtime <= 0 or candidate_runtime <= 0:
                raise ValueError("runtime_seconds must be positive")
            for label, metadata in (("reference", reference_metadata), ("candidate", candidate_metadata)):
                if int(metadata.get("samples", -1)) != EXPECTED_SAMPLES:
                    raise ValueError(f"{label} metadata samples={metadata.get('samples')!r}")
                if int(metadata.get("steps", -1)) != EXPECTED_STEPS:
                    raise ValueError(f"{label} metadata steps={metadata.get('steps')!r}")
            evaluation = read_evaluation(evaluation_path, dataset)
            expected_methods = {"B", "G", variant}
            if set(evaluation) != expected_methods:
                raise ValueError(f"evaluator methods={sorted(evaluation)}, expected {sorted(expected_methods)}")
            runs[key] = {
                "path": path,
                "provenance": values,
                "paired_hash": read_paired_hash(path / "unconditional_sha256.txt"),
                "reference_metadata": reference_metadata,
                "candidate_metadata": candidate_metadata,
                "reference_runtime": reference_runtime,
                "candidate_runtime": candidate_runtime,
                "evaluation": evaluation,
                "mechanism": read_dense_mechanism(mechanism_path),
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    expected = {(dataset, variant) for dataset in DATASETS for variant in WITNESS_VARIANTS}
    missing = sorted(expected - set(runs))
    extra = sorted(set(runs) - expected)
    if missing:
        errors.append(f"missing runs: {missing}")
    if extra:
        errors.append(f"unexpected runs: {extra}")
    if len(paths) != len(expected):
        errors.append(f"matched {len(paths)} run directories; expected {len(expected)}")
    if errors:
        raise ValueError("incomplete/invalid CFRW round:\n- " + "\n- ".join(errors))
    return runs


def evaluate_candidate(runs: Mapping[tuple[str, str], Mapping[str, Any]], variant: str) -> dict[str, Any]:
    reasons: list[str] = []
    per_dataset: dict[str, Any] = {}
    primary_g: list[float] = []
    primary_b: list[float] = []
    alpha_g: list[float] = []
    runtime_ratios: list[float] = []
    for dataset in DATASETS:
        run = runs[(dataset, variant)]
        b_local = oriented(dataset, run["evaluation"]["B"])
        candidate_value = oriented(dataset, run["evaluation"][variant])
        g_value = oriented(dataset, run["evaluation"]["G"])
        candidate_vs_b = difference(candidate_value, b_local)
        g_vs_b = difference(g_value, b_local)
        candidate_vs_g = difference(candidate_value, g_value)
        mechanism_reasons = inspect_witness(run["candidate_metadata"], run["mechanism"], variant)
        reasons.extend(f"{dataset}: {reason}" for reason in mechanism_reasons)

        runtime_ratio = float(run["candidate_runtime"]) / float(run["reference_runtime"])
        runtime_ratios.append(runtime_ratio)
        if runtime_ratio > 3.0 + EPSILON:
            reasons.append(f"{dataset}: runtime ratio {runtime_ratio:.3f} exceeds 3x G")

        for comparator, delta in (("G", candidate_vs_g), ("B", candidate_vs_b)):
            if delta["shape"] < -0.05 - EPSILON:
                reasons.append(f"{dataset}: Shape benefit vs {comparator} {delta['shape']:.6f} < -0.05")
            if delta["trend"] < -0.05 - EPSILON:
                reasons.append(f"{dataset}: Trend benefit vs {comparator} {delta['trend']:.6f} < -0.05")
            if delta["alpha"] < -0.10 - EPSILON:
                reasons.append(f"{dataset}: Alpha benefit vs {comparator} {delta['alpha']:.6f} < -0.10")
            if delta["beta"] < -0.15 - EPSILON:
                reasons.append(f"{dataset}: Beta benefit vs {comparator} {delta['beta']:.6f} < -0.15")
            if dataset == "news":
                if delta["r2"] < -0.10 - EPSILON:
                    reasons.append(f"news: R2 benefit vs {comparator} {delta['r2']:.6f} < -0.10")
                if delta["nrmse"] < -0.05 - EPSILON:
                    reasons.append(f"news: NRMSE benefit vs {comparator} {delta['nrmse']:.6f} < -0.05")
            elif delta["auc"] < -0.005 - EPSILON:
                reasons.append(f"default: AUC benefit vs {comparator} {delta['auc']:.6f} < -0.005")

        if max(candidate_vs_g["shape"], candidate_vs_g["trend"]) <= EPSILON:
            reasons.append(f"{dataset}: improves neither Shape nor Trend over G")
        primary_g.extend((candidate_vs_g["shape"], candidate_vs_g["trend"]))
        primary_b.extend((candidate_vs_b["shape"], candidate_vs_b["trend"]))
        alpha_g.append(candidate_vs_g["alpha"])
        per_dataset[dataset] = {
            "candidate_raw": run["evaluation"][variant],
            "candidate_vs_b_oriented": candidate_vs_b,
            "g_vs_b_oriented": g_vs_b,
            "candidate_vs_g_oriented": candidate_vs_g,
            "reference_runtime_seconds": run["reference_runtime"],
            "candidate_runtime_seconds": run["candidate_runtime"],
            "runtime_ratio_to_g": runtime_ratio,
            "paired_unconditional_sha256": run["paired_hash"],
            "mechanism": run["mechanism"],
        }

    shape_g = [per_dataset[dataset]["candidate_vs_g_oriented"]["shape"] for dataset in DATASETS]
    trend_g = [per_dataset[dataset]["candidate_vs_g_oriented"]["trend"] for dataset in DATASETS]
    shape_b = [per_dataset[dataset]["candidate_vs_b_oriented"]["shape"] for dataset in DATASETS]
    trend_b = [per_dataset[dataset]["candidate_vs_b_oriented"]["trend"] for dataset in DATASETS]
    for label, values in (("mean Shape vs G", shape_g), ("mean Trend vs G", trend_g), ("mean Shape vs B", shape_b), ("mean Trend vs B", trend_b)):
        if statistics.fmean(values) <= EPSILON:
            reasons.append(f"{label} must be positive; found {statistics.fmean(values):.6f}")
    primary_wins_g = sum(delta > EPSILON for delta in primary_g)
    if primary_wins_g < 3:
        reasons.append(f"requires >=3/4 Shape/Trend wins vs G, found {primary_wins_g}/4")

    return {
        "variant": variant,
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "primary_wins_vs_g": primary_wins_g,
        "mean_shape_benefit_vs_g": statistics.fmean(shape_g),
        "mean_trend_benefit_vs_g": statistics.fmean(trend_g),
        "mean_shape_benefit_vs_b": statistics.fmean(shape_b),
        "mean_trend_benefit_vs_b": statistics.fmean(trend_b),
        "mean_alpha_benefit_vs_g": statistics.fmean(alpha_g),
        "mean_runtime_ratio": statistics.fmean(runtime_ratios),
        "per_dataset": per_dataset,
    }


def format_number(value: object) -> str:
    number = float(value)
    if abs(number) < 0.5e-6:
        number = 0.0
    return f"{number:.6f}"


def write_summary(output: Path, summaries: Sequence[Mapping[str, Any]], winners: Sequence[str], runs: Mapping[tuple[str, str], Mapping[str, Any]]) -> None:
    headers = ["Rank", "Selected", "Arm", "Eligible", "Primary wins/G", "mean dShape/G", "mean dTrend/G", "mean dShape/B", "mean dTrend/B", "mean dAlpha/G", "runtime x"]
    rows = []
    for summary in summaries:
        rows.append([
            summary["rank"], "yes" if summary["selected"] else "", summary["variant"],
            "yes" if summary["eligible"] else "no", f"{summary['primary_wins_vs_g']}/4",
            format_number(summary["mean_shape_benefit_vs_g"]), format_number(summary["mean_trend_benefit_vs_g"]),
            format_number(summary["mean_shape_benefit_vs_b"]), format_number(summary["mean_trend_benefit_vs_b"]),
            format_number(summary["mean_alpha_benefit_vs_g"]), f"{float(summary['mean_runtime_ratio']):.3f}",
        ])
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    table.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    hashes = [
        f"- {dataset}/{variant} paired unconditional SHA-256: `{runs[(dataset, variant)]['paired_hash']}`"
        for dataset in DATASETS for variant in WITNESS_VARIANTS
    ]
    lines = [
        "# CFRW two-dataset validation selection", "",
        "- Design: **News + Default x W0/W4/W8/W12**, one locked validation seed per dataset.",
        "- Each of the eight GPU tasks sampled its locked G reference and candidate sequentially.",
        "- Pairing: **passed**; reference and candidate unconditional SHA-256 match inside every task.",
        *hashes,
        f"- Selected (maximum two): **{', '.join(winners) if winners else 'none'}**", "",
        "Benefit deltas are oriented so positive is better and are computed from one file-local B/G/W evaluation.", "",
        *table, "", "## Guardrails", "",
        "- Exact eight paired runs, declared seeds, 8192 samples, 50 steps, and matching reference/candidate unconditional hash in every task.",
        "- CFRW must be train-only, preserve the target, fire exactly once, select rows/cells, and retain the requested factor families.",
        "- CFRW runtime must be at most 3x its paired G reference on the same task.",
        "- Per dataset/comparator downside floors: Shape/Trend -0.05 pp, Alpha -0.10 pp, Beta -0.15 pp.",
        "- News utility floors: R2 -0.10 and NRMSE benefit -0.05; Default AUC floor: -0.005.",
        "- Mean Shape and Trend benefit must each be positive vs both G and B; at least 3/4 dataset x primary-metric comparisons must beat G.", "",
        "## Rejections", "",
    ]
    rejected = [summary for summary in summaries if not summary["eligible"]]
    if rejected:
        for summary in rejected:
            lines.append(f"- **{summary['variant']}**: " + "; ".join(summary["rejection_reasons"]))
    else:
        lines.append("- None.")
    lines.append("")
    (output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    fields = ["rank", "selected", "eligible", "variant", "primary_wins_vs_g", "mean_shape_benefit_vs_g", "mean_trend_benefit_vs_g", "mean_shape_benefit_vs_b", "mean_trend_benefit_vs_b", "mean_alpha_benefit_vs_g", "mean_runtime_ratio", "rejection_reasons"]
    with (output / "SUMMARY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: "; ".join(summary[key]) if key == "rejection_reasons" else summary[key] for key in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", action="append", required=True, help="Glob matching CFRW run directories; may be repeated.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-winners", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (0 <= args.max_winners <= 2):
        raise ValueError("--max-winners must be in [0,2]")
    paths = discover_run_dirs(args.input_glob)
    runs = load_runs(paths)

    candidates = [evaluate_candidate(runs, variant) for variant in WITNESS_VARIANTS]
    ranked = sorted(candidates, key=lambda item: (
        not item["eligible"], -item["primary_wins_vs_g"],
        -item["mean_shape_benefit_vs_g"], -item["mean_trend_benefit_vs_g"],
        -item["mean_alpha_benefit_vs_g"], item["mean_runtime_ratio"], item["variant"],
    ))
    winners: list[str] = []
    for rank, summary in enumerate(ranked, start=1):
        summary["rank"] = rank
        summary["selected"] = bool(summary["eligible"] and len(winners) < args.max_winners)
        if summary["selected"]:
            winners.append(str(summary["variant"]))

    args.output.mkdir(parents=True, exist_ok=False)
    write_summary(args.output, ranked, winners, runs)
    selection = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "round_complete": True,
        "datasets": list(DATASETS),
        "variants": list(WITNESS_VARIANTS),
        "seeds": EXPECTED_SEEDS,
        "samples": EXPECTED_SAMPLES,
        "steps": EXPECTED_STEPS,
        "max_winners": args.max_winners,
        "winners": winners,
        "candidates": ranked,
        "input_directories": [str(path) for path in paths],
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print((args.output / "SUMMARY.md").resolve())
    print(f"Selected {len(winners)} arm(s): {', '.join(winners) or 'none'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
