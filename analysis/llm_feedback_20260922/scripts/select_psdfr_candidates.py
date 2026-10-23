#!/usr/bin/env python3
"""Select News PS-DFR candidates from paired evaluator CSV files.

The screening round is expected to contain six candidate methods evaluated at
two seeds.  Every ``metrics.csv`` must contain a locked ``Reference`` PS-DFR
row, a raw ``Backbone`` control, and one named candidate row.  Pairing is
deliberately file-local so that GPU-class and sampler differences cannot leak
into a candidate-versus-reference delta.

All ``*_oriented_delta`` values use the convention "positive is better".  Raw
deltas retain the usual ``candidate - Reference`` convention, so a negative raw
NRMSE delta is an improvement.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


BASELINE_METHOD = "Reference"
IGNORED_CONTROL_METHODS = {"Backbone"}
EXPECTED_VARIANTS = 6
EXPECTED_PAIRS = 2
EPSILON = 1e-12


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool
    aliases: tuple[str, ...]


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "shape",
        "Shape",
        True,
        (
            "shape",
            "shape_score",
            "shape_score_percent",
            "shape_percent",
            "column_shapes",
            "column_shape_score",
            "column_shape_score_percent",
        ),
    ),
    MetricSpec(
        "trend",
        "Trend",
        True,
        (
            "trend",
            "trend_score",
            "trend_score_percent",
            "trend_percent",
            "column_pair_trends",
            "column_pair_trend_score",
            "column_pair_trend_score_percent",
        ),
    ),
    MetricSpec(
        "alpha",
        "Alpha",
        True,
        (
            "alpha",
            "alpha_precision",
            "alpha_precision_percent",
            "alpha_score",
            "alpha_score_percent",
        ),
    ),
    MetricSpec(
        "beta",
        "Beta",
        True,
        (
            "beta",
            "beta_recall",
            "beta_recall_percent",
            "beta_score",
            "beta_score_percent",
        ),
    ),
    MetricSpec(
        "r2",
        "R2",
        True,
        (
            "r2",
            "r_squared",
            "mle_r2",
            "mle_xgboost_r2",
            "xgboost_regression_r2",
            "tstr_r2",
            "utility_r2",
        ),
    ),
    MetricSpec(
        "nrmse",
        "NRMSE",
        False,
        (
            "nrmse",
            "mle_nrmse",
            "mle_xgboost_nrmse",
            "xgboost_regression_nrmse",
            "tstr_nrmse",
            "utility_nrmse",
        ),
    ),
)

METRIC_BY_KEY = {metric.key: metric for metric in METRICS}

# Guardrails are intentionally expressed in the evaluator's native units.
# Shape/Trend/Alpha/Beta are percentage-point scores in the News evaluator.
MEAN_DOWNSIDE_FLOORS = {
    "beta": -0.10,
    "r2": -0.10,
    "nrmse": -0.05,
}


def normalize_name(value: str) -> str:
    value = value.strip().casefold().replace("²", "2")
    return re.sub(r"[^a-z0-9]+", "", value)


def canonical_method(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def parse_float(value: object, *, context: str) -> float:
    text = "" if value is None else str(value).strip()
    text = text.replace("−", "-").replace("–", "-")
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        parsed = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric value for {context}, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Expected a finite value for {context}, got {value!r}")
    return parsed


def resolve_column(fieldnames: Sequence[str], aliases: Iterable[str]) -> str | None:
    normalized = {normalize_name(name): name for name in fieldnames}
    matches = [normalized[normalize_name(alias)] for alias in aliases if normalize_name(alias) in normalized]
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) > 1:
        raise ValueError(f"Ambiguous columns for aliases {tuple(aliases)!r}: {unique_matches}")
    return unique_matches[0] if unique_matches else None


def discover_seed(csv_path: Path, rows: Sequence[Mapping[str, str]], fieldnames: Sequence[str]) -> str:
    seed_column = resolve_column(fieldnames, ("seed", "random_seed", "sample_seed"))
    if seed_column:
        seeds = {str(row.get(seed_column, "")).strip() for row in rows if str(row.get(seed_column, "")).strip()}
        if len(seeds) > 1:
            raise ValueError(f"{csv_path}: multiple seeds in one paired metrics file: {sorted(seeds)}")
        if seeds:
            return next(iter(seeds))

    directory = csv_path.parent
    for _ in range(5):
        provenance = directory / "provenance.txt"
        if provenance.is_file():
            for line in provenance.read_text(encoding="utf-8", errors="replace").splitlines():
                match = re.fullmatch(r"\s*seed\s*=\s*(\S+)\s*", line, flags=re.IGNORECASE)
                if match:
                    return match.group(1)
        if directory.parent == directory:
            break
        directory = directory.parent

    path_text = str(csv_path)
    match = re.search(r"(?:^|[\\/_-])seed[_=-]?(\d+)(?:$|[\\/_.-])", path_text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(?:^|[\\/_-])s(\d{4,})(?:$|[\\/_.-])", path_text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return f"file:{csv_path.as_posix()}"


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match).resolve()
            if path.is_file():
                paths[str(path).casefold()] = path
    return sorted(paths.values(), key=lambda path: str(path).casefold())


def load_pairs(csv_paths: Sequence[Path]) -> tuple[list[dict[str, object]], dict[str, str]]:
    pairs: list[dict[str, object]] = []
    display_names: dict[str, str] = {}

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        if not rows:
            raise ValueError(f"{csv_path}: metrics CSV is empty")
        method_column = resolve_column(fieldnames, ("method",))
        if method_column is None:
            raise ValueError(f"{csv_path}: missing required 'method' column; found {fieldnames}")

        metric_columns: dict[str, str] = {}
        for metric in METRICS:
            column = resolve_column(fieldnames, metric.aliases)
            if column is None:
                raise ValueError(
                    f"{csv_path}: cannot resolve News metric {metric.label!r}; "
                    f"found columns {fieldnames}"
                )
            metric_columns[metric.key] = column

        indexed_rows: dict[str, Mapping[str, str]] = {}
        row_names: dict[str, str] = {}
        for row in rows:
            display_name = str(row.get(method_column, "")).strip()
            if not display_name:
                raise ValueError(f"{csv_path}: blank method name")
            canonical = canonical_method(display_name)
            if canonical in indexed_rows:
                raise ValueError(f"{csv_path}: duplicate method row {display_name!r}")
            indexed_rows[canonical] = row
            row_names[canonical] = display_name

        baseline_key = canonical_method(BASELINE_METHOD)
        if baseline_key not in indexed_rows:
            raise ValueError(
                f"{csv_path}: missing {BASELINE_METHOD!r} row; "
                f"found {sorted(row_names.values())}"
            )

        seed = discover_seed(csv_path, rows, fieldnames)
        baseline_row = indexed_rows[baseline_key]
        baseline_values = {
            key: parse_float(baseline_row[column], context=f"{csv_path}:{BASELINE_METHOD}:{column}")
            for key, column in metric_columns.items()
        }

        for method_key, candidate_row in indexed_rows.items():
            if method_key == baseline_key or method_key in {
                canonical_method(name) for name in IGNORED_CONTROL_METHODS
            }:
                continue
            display_name = row_names[method_key]
            display_names.setdefault(method_key, display_name)
            candidate_values = {
                key: parse_float(candidate_row[column], context=f"{csv_path}:{display_name}:{column}")
                for key, column in metric_columns.items()
            }
            raw_deltas = {
                key: candidate_values[key] - baseline_values[key]
                for key in METRIC_BY_KEY
            }
            oriented_deltas = {
                key: raw_deltas[key] if METRIC_BY_KEY[key].higher_is_better else -raw_deltas[key]
                for key in METRIC_BY_KEY
            }
            pairs.append(
                {
                    "method_key": method_key,
                    "method": display_name,
                    "seed": seed,
                    "source": str(csv_path),
                    "baseline": baseline_values,
                    "candidate": candidate_values,
                    "raw_delta": raw_deltas,
                    "oriented_delta": oriented_deltas,
                }
            )

    if not pairs:
        raise ValueError(f"No candidate rows were found beside {BASELINE_METHOD}")
    return pairs, display_names


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_candidates(
    pairs: Sequence[Mapping[str, object]], display_names: Mapping[str, str]
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["method_key"])].append(pair)

    summaries: list[dict[str, object]] = []
    for method_key in sorted(grouped):
        method_pairs = grouped[method_key]
        seeds = [str(pair["seed"]) for pair in method_pairs]
        reasons: list[str] = []
        if len(method_pairs) != EXPECTED_PAIRS:
            reasons.append(f"requires exactly {EXPECTED_PAIRS} paired seeds, found {len(method_pairs)}")
        if len(set(seeds)) != len(seeds):
            reasons.append(f"duplicate seed labels: {seeds}")

        metric_summary: dict[str, dict[str, float | int]] = {}
        for metric in METRICS:
            key = metric.key
            baseline_values = [float(pair["baseline"][key]) for pair in method_pairs]  # type: ignore[index]
            candidate_values = [float(pair["candidate"][key]) for pair in method_pairs]  # type: ignore[index]
            raw_deltas = [float(pair["raw_delta"][key]) for pair in method_pairs]  # type: ignore[index]
            oriented_deltas = [float(pair["oriented_delta"][key]) for pair in method_pairs]  # type: ignore[index]
            metric_summary[key] = {
                "baseline_mean": statistics.fmean(baseline_values),
                "candidate_mean": statistics.fmean(candidate_values),
                "raw_delta_mean": statistics.fmean(raw_deltas),
                "raw_delta_std": sample_std(raw_deltas),
                "oriented_delta_mean": statistics.fmean(oriented_deltas),
                "oriented_delta_std": sample_std(oriented_deltas),
                "wins": sum(delta > EPSILON for delta in oriented_deltas),
                "ties": sum(abs(delta) <= EPSILON for delta in oriented_deltas),
                "losses": sum(delta < -EPSILON for delta in oriented_deltas),
                "win_rate": sum(delta > EPSILON for delta in oriented_deltas) / len(oriented_deltas),
            }

        for key in ("shape", "trend"):
            wins = int(metric_summary[key]["wins"])
            if wins != len(method_pairs):
                reasons.append(f"{METRIC_BY_KEY[key].label} must improve in every paired seed ({wins}/{len(method_pairs)})")
        if float(metric_summary["alpha"]["oriented_delta_mean"]) <= EPSILON:
            reasons.append("Alpha mean delta must be positive")
        if float(metric_summary["alpha"]["win_rate"]) < 0.5:
            reasons.append("Alpha must win at least half of paired seeds")

        mean_metric_wins = sum(
            float(metric_summary[metric.key]["oriented_delta_mean"]) > EPSILON for metric in METRICS
        )
        paired_wins = sum(int(metric_summary[metric.key]["wins"]) for metric in METRICS)
        paired_comparisons = len(method_pairs) * len(METRICS)
        paired_win_rate = paired_wins / paired_comparisons
        if mean_metric_wins < 4:
            reasons.append(f"requires positive mean delta on >=4/6 metrics, found {mean_metric_wins}/6")
        if paired_win_rate + EPSILON < (2.0 / 3.0):
            reasons.append(f"paired metric win rate must be >=2/3, found {paired_win_rate:.3f}")

        for key, floor in MEAN_DOWNSIDE_FLOORS.items():
            value = float(metric_summary[key]["oriented_delta_mean"])
            if value < floor - EPSILON:
                reasons.append(
                    f"{METRIC_BY_KEY[key].label} mean benefit-oriented delta {value:.6g} "
                    f"breaches floor {floor:.6g}"
                )

        primary_consistency = statistics.fmean(
            float(metric_summary[key]["win_rate"]) for key in ("shape", "trend", "alpha")
        )
        summaries.append(
            {
                "method_key": method_key,
                "method": display_names[method_key],
                "pair_count": len(method_pairs),
                "seeds": seeds,
                "eligible": not reasons,
                "rejection_reasons": reasons,
                "mean_metric_wins": mean_metric_wins,
                "paired_wins": paired_wins,
                "paired_comparisons": paired_comparisons,
                "paired_win_rate": paired_win_rate,
                "primary_consistency": primary_consistency,
                "metrics": metric_summary,
            }
        )

    # Normalize effects per metric across the six arms so large-scale Alpha or
    # R2 values cannot dominate Shape/Trend.  The score is only a tie-breaker;
    # hard guardrails and sign consistency decide eligibility first.
    scales: dict[str, float] = {}
    for metric in METRICS:
        effects = [abs(float(summary["metrics"][metric.key]["oriented_delta_mean"])) for summary in summaries]  # type: ignore[index]
        scale = statistics.median(effects)
        if scale <= EPSILON:
            scale = max(effects, default=1.0)
        scales[metric.key] = scale if scale > EPSILON else 1.0

    for summary in summaries:
        normalized_effects = []
        for metric in METRICS:
            effect = float(summary["metrics"][metric.key]["oriented_delta_mean"])  # type: ignore[index]
            normalized_effects.append(max(-3.0, min(3.0, effect / scales[metric.key])))
        summary["balanced_effect_score"] = statistics.fmean(normalized_effects)

    return summaries


def rank_and_select(
    summaries: list[dict[str, object]], *, max_winners: int
) -> tuple[list[dict[str, object]], bool, list[str]]:
    round_warnings: list[str] = []
    if len(summaries) != EXPECTED_VARIANTS:
        round_warnings.append(
            f"Expected {EXPECTED_VARIANTS} candidate variants, found {len(summaries)}; "
            "the round is incomplete and no arm will be promoted."
        )
    incomplete_pairs = [
        str(summary["method"])
        for summary in summaries
        if int(summary["pair_count"]) != EXPECTED_PAIRS
    ]
    if incomplete_pairs:
        round_warnings.append(
            f"Variants without exactly {EXPECTED_PAIRS} pairs: {', '.join(incomplete_pairs)}; "
            "the round is incomplete and no arm will be promoted."
        )

    round_complete = not round_warnings
    ranked = sorted(
        summaries,
        key=lambda summary: (
            not bool(summary["eligible"]),
            -float(summary["primary_consistency"]),
            -int(summary["mean_metric_wins"]),
            -float(summary["paired_win_rate"]),
            -float(summary["balanced_effect_score"]),
            str(summary["method"]).casefold(),
        ),
    )
    winners: list[dict[str, object]] = []
    for rank, summary in enumerate(ranked, start=1):
        summary["rank"] = rank
        summary["selected"] = False
        if round_complete and bool(summary["eligible"]) and len(winners) < max_winners:
            summary["selected"] = True
            winners.append(summary)
    return winners, round_complete, round_warnings


def format_number(value: object, digits: int = 6) -> str:
    number = float(value)
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_paired_csv(output_path: Path, pairs: Sequence[Mapping[str, object]]) -> None:
    fields = ["method", "seed", "source"]
    for metric in METRICS:
        fields.extend(
            (
                f"backbone_{metric.key}",
                f"candidate_{metric.key}",
                f"raw_delta_{metric.key}",
                f"oriented_delta_{metric.key}",
            )
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in sorted(pairs, key=lambda item: (str(item["method"]).casefold(), str(item["seed"]))):
            row: dict[str, object] = {
                "method": pair["method"],
                "seed": pair["seed"],
                "source": pair["source"],
            }
            for metric in METRICS:
                row[f"backbone_{metric.key}"] = pair["baseline"][metric.key]  # type: ignore[index]
                row[f"candidate_{metric.key}"] = pair["candidate"][metric.key]  # type: ignore[index]
                row[f"raw_delta_{metric.key}"] = pair["raw_delta"][metric.key]  # type: ignore[index]
                row[f"oriented_delta_{metric.key}"] = pair["oriented_delta"][metric.key]  # type: ignore[index]
            writer.writerow(row)


def write_summary_csv(output_path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "rank",
        "selected",
        "eligible",
        "method",
        "pair_count",
        "seeds",
        "mean_metric_wins",
        "paired_wins",
        "paired_comparisons",
        "paired_win_rate",
        "primary_consistency",
        "balanced_effect_score",
        "rejection_reasons",
    ]
    for metric in METRICS:
        fields.extend(
            (
                f"backbone_mean_{metric.key}",
                f"candidate_mean_{metric.key}",
                f"raw_delta_mean_{metric.key}",
                f"raw_delta_std_{metric.key}",
                f"oriented_delta_mean_{metric.key}",
                f"oriented_delta_std_{metric.key}",
                f"wins_{metric.key}",
                f"ties_{metric.key}",
                f"losses_{metric.key}",
                f"win_rate_{metric.key}",
            )
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row: dict[str, object] = {
                "rank": summary["rank"],
                "selected": summary["selected"],
                "eligible": summary["eligible"],
                "method": summary["method"],
                "pair_count": summary["pair_count"],
                "seeds": ";".join(str(seed) for seed in summary["seeds"]),  # type: ignore[union-attr]
                "mean_metric_wins": summary["mean_metric_wins"],
                "paired_wins": summary["paired_wins"],
                "paired_comparisons": summary["paired_comparisons"],
                "paired_win_rate": summary["paired_win_rate"],
                "primary_consistency": summary["primary_consistency"],
                "balanced_effect_score": summary["balanced_effect_score"],
                "rejection_reasons": "; ".join(summary["rejection_reasons"]),  # type: ignore[arg-type]
            }
            for metric in METRICS:
                values = summary["metrics"][metric.key]  # type: ignore[index]
                row.update(
                    {
                        f"backbone_mean_{metric.key}": values["baseline_mean"],
                        f"candidate_mean_{metric.key}": values["candidate_mean"],
                        f"raw_delta_mean_{metric.key}": values["raw_delta_mean"],
                        f"raw_delta_std_{metric.key}": values["raw_delta_std"],
                        f"oriented_delta_mean_{metric.key}": values["oriented_delta_mean"],
                        f"oriented_delta_std_{metric.key}": values["oriented_delta_std"],
                        f"wins_{metric.key}": values["wins"],
                        f"ties_{metric.key}": values["ties"],
                        f"losses_{metric.key}": values["losses"],
                        f"win_rate_{metric.key}": values["win_rate"],
                    }
                )
            writer.writerow(row)


def write_summary_markdown(
    output_path: Path,
    summaries: Sequence[Mapping[str, object]],
    winners: Sequence[Mapping[str, object]],
    csv_paths: Sequence[Path],
    round_complete: bool,
    round_warnings: Sequence[str],
) -> None:
    headers = [
        "Rank",
        "Selected",
        "Method",
        "Pairs",
        "Eligible",
        "Mean wins",
        "Paired wins",
        "Δ Shape",
        "Δ Trend",
        "Δ Alpha",
        "Δ Beta",
        "Δ R2",
        "Δ NRMSE+",
    ]
    rows = []
    for summary in summaries:
        metrics = summary["metrics"]
        rows.append(
            (
                summary["rank"],
                "yes" if summary["selected"] else "",
                summary["method"],
                summary["pair_count"],
                "yes" if summary["eligible"] else "no",
                f"{summary['mean_metric_wins']}/6",
                f"{summary['paired_wins']}/{summary['paired_comparisons']}",
                format_number(metrics["shape"]["oriented_delta_mean"]),
                format_number(metrics["trend"]["oriented_delta_mean"]),
                format_number(metrics["alpha"]["oriented_delta_mean"]),
                format_number(metrics["beta"]["oriented_delta_mean"]),
                format_number(metrics["r2"]["oriented_delta_mean"]),
                format_number(metrics["nrmse"]["oriented_delta_mean"]),
            )
        )

    lines = [
        "# PS-DFR candidate selection",
        "",
        f"- Input files: **{len(csv_paths)}**",
        f"- Expected design: **{EXPECTED_VARIANTS} variants × {EXPECTED_PAIRS} paired seeds**",
        f"- Round complete: **{'yes' if round_complete else 'no'}**",
        f"- Selected: **{', '.join(str(winner['method']) for winner in winners) if winners else 'none'}**",
        "- Delta convention in the table: benefit-oriented; positive is better. "
        "For `Δ NRMSE+`, this is `Reference - candidate`.",
        "",
    ]
    if round_warnings:
        lines.extend(["## Round warnings", ""])
        lines.extend(f"- {warning}" for warning in round_warnings)
        lines.append("")

    lines.extend(["## Ranking", "", markdown_table(headers, rows), "", "## Guardrails", ""])
    lines.extend(
        [
            f"- Exactly {EXPECTED_PAIRS} file-local Reference pairs per arm and {EXPECTED_VARIANTS} arms in the round.",
            "- Shape and Trend must improve in every paired seed.",
            "- Mean Alpha delta must be positive and Alpha must win at least half the seeds.",
            "- At least 4/6 metrics must have positive mean deltas.",
            "- At least two thirds of all seed × metric comparisons must be wins.",
            "- Mean downside floors (benefit-oriented): Beta ≥ -0.10, R2 ≥ -0.10, NRMSE+ ≥ -0.05.",
            "- Ranking is lexicographic: primary consistency, mean metric wins, paired win rate, then balanced effect score.",
            "",
            "## Rejections",
            "",
        ]
    )
    rejected = [summary for summary in summaries if not summary["eligible"]]
    if rejected:
        for summary in rejected:
            reasons = "; ".join(str(reason) for reason in summary["rejection_reasons"])
            lines.append(f"- **{summary['method']}**: {reasons}")
    else:
        lines.append("- None.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def jsonable_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "rank": summary["rank"],
        "selected": summary["selected"],
        "eligible": summary["eligible"],
        "method": summary["method"],
        "pair_count": summary["pair_count"],
        "seeds": summary["seeds"],
        "mean_metric_wins": summary["mean_metric_wins"],
        "paired_wins": summary["paired_wins"],
        "paired_comparisons": summary["paired_comparisons"],
        "paired_win_rate": summary["paired_win_rate"],
        "primary_consistency": summary["primary_consistency"],
        "balanced_effect_score": summary["balanced_effect_score"],
        "rejection_reasons": summary["rejection_reasons"],
        "metrics": summary["metrics"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate paired News metrics.csv files, compare against locked PS-DFR, "
            "promotion guardrails, and select at most two candidates."
        )
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        required=True,
        help="Glob for metrics.csv inputs; may be repeated and supports **.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--max-winners", type=int, default=2, help="Maximum promoted arms (default: 2).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_winners < 0:
        raise ValueError("--max-winners must be non-negative")

    csv_paths = expand_inputs(args.input_glob)
    if not csv_paths:
        raise FileNotFoundError(f"No files matched --input-glob values: {args.input_glob}")

    pairs, display_names = load_pairs(csv_paths)
    summaries = summarize_candidates(pairs, display_names)
    winners, round_complete, round_warnings = rank_and_select(
        summaries, max_winners=args.max_winners
    )
    ranked = sorted(summaries, key=lambda summary: int(summary["rank"]))

    args.output.mkdir(parents=True, exist_ok=True)
    write_paired_csv(args.output / "PAIRED_DELTAS.csv", pairs)
    write_summary_csv(args.output / "SUMMARY.csv", ranked)
    write_summary_markdown(
        args.output / "SUMMARY.md",
        ranked,
        winners,
        csv_paths,
        round_complete,
        round_warnings,
    )

    selection = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_method": BASELINE_METHOD,
        "input_files": [str(path) for path in csv_paths],
        "expected_variants": EXPECTED_VARIANTS,
        "expected_pairs_per_variant": EXPECTED_PAIRS,
        "round_complete": round_complete,
        "round_warnings": list(round_warnings),
        "max_winners": args.max_winners,
        "winners": [str(winner["method"]) for winner in winners],
        "delta_convention": {
            "raw_delta": "candidate - Reference",
            "oriented_delta": "positive is better; Reference - candidate for NRMSE",
        },
        "guardrails": {
            "shape_and_trend": "must improve in every paired seed",
            "alpha": "positive mean and >=50% paired wins",
            "mean_metric_wins_minimum": 4,
            "paired_win_rate_minimum": 2.0 / 3.0,
            "mean_downside_floors_oriented": MEAN_DOWNSIDE_FLOORS,
        },
        "candidates": [jsonable_summary(summary) for summary in ranked],
    }
    (args.output / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print((args.output / "SUMMARY.md").resolve())
    print(f"Selected {len(winners)} arm(s): {', '.join(selection['winners']) or 'none'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
