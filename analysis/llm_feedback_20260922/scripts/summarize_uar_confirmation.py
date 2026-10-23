#!/usr/bin/env python3
"""Summarize paper-grade UAR confirmation runs on News.

Each input ``metrics.csv`` must contain exactly the four controls ``Backbone``,
``Global``, ``Frozen`` and ``Reference``, plus one selected candidate.  All
comparisons are paired within a file/seed.  A candidate is decision-eligible
only with six distinct, identifiable seeds; incomplete or duplicate-seed
evidence is retained in the reports but fails closed.
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


EXPECTED_SEEDS = 6
CONTROL_NAMES = ("Reference", "Global", "Frozen", "Backbone")
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
CONTROL_BY_KEY = {name.casefold(): name for name in CONTROL_NAMES}

# Benefit-oriented deltas are positive when the candidate is better.  These
# floors therefore cap acceptable mean damage relative to every comparator.
CATASTROPHIC_FLOORS = {
    "beta": -0.10,
    "r2": -0.10,
    "nrmse": -0.05,
}


def normalize_name(value: str) -> str:
    value = value.strip().casefold().replace("²", "2")
    return re.sub(r"[^a-z0-9]+", "", value)


def canonical_method(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def resolve_column(fieldnames: Sequence[str], aliases: Iterable[str]) -> str | None:
    normalized: dict[str, list[str]] = defaultdict(list)
    for name in fieldnames:
        normalized[normalize_name(name)].append(name)
    matches: list[str] = []
    for alias in aliases:
        matches.extend(normalized.get(normalize_name(alias), []))
    unique = list(dict.fromkeys(matches))
    if len(unique) > 1:
        raise ValueError(f"Ambiguous columns for aliases {tuple(aliases)!r}: {unique}")
    return unique[0] if unique else None


def parse_float(value: object, *, context: str) -> float:
    text = "" if value is None else str(value).strip()
    text = text.replace("−", "-").replace("–", "-")
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric {context}, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Expected finite {context}, got {value!r}")
    return number


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match).resolve()
            if path.is_file():
                paths[str(path).casefold()] = path
    return sorted(paths.values(), key=lambda item: str(item).casefold())


def discover_seed(
    csv_path: Path,
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
) -> str:
    seed_column = resolve_column(fieldnames, ("seed", "random_seed", "sample_seed"))
    if seed_column:
        seeds = {
            str(row.get(seed_column, "")).strip()
            for row in rows
            if str(row.get(seed_column, "")).strip()
        }
        if len(seeds) > 1:
            raise ValueError(f"{csv_path}: multiple seeds in one metrics file: {sorted(seeds)}")
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
    for pattern in (
        r"(?:^|[\\/_-])seed[_=-]?(\d+)(?:$|[\\/_.-])",
        r"(?:^|[\\/_-])s(\d{4,})(?:$|[\\/_.-])",
    ):
        match = re.search(pattern, path_text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    raise ValueError(
        f"{csv_path}: cannot identify seed from a seed column, provenance.txt, or path; "
        "confirmation fails closed"
    )


def read_values(
    csv_path: Path,
    method: str,
    row: Mapping[str, str],
    metric_columns: Mapping[str, str],
) -> dict[str, float]:
    return {
        key: parse_float(row[column], context=f"{csv_path}:{method}:{column}")
        for key, column in metric_columns.items()
    }


def load_comparisons(
    csv_paths: Sequence[Path],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    comparisons: list[dict[str, object]] = []
    candidate_names: dict[str, str] = {}

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

        indexed: dict[str, Mapping[str, str]] = {}
        display_names: dict[str, str] = {}
        for row in rows:
            display = str(row.get(method_column, "")).strip()
            if not display:
                raise ValueError(f"{csv_path}: blank method name")
            key = canonical_method(display)
            if key in indexed:
                raise ValueError(f"{csv_path}: duplicate method row {display!r}")
            indexed[key] = row
            display_names[key] = display

        missing_controls = [
            control for control in CONTROL_NAMES if canonical_method(control) not in indexed
        ]
        if missing_controls:
            raise ValueError(
                f"{csv_path}: missing controls {missing_controls}; "
                f"found {sorted(display_names.values())}"
            )
        control_keys = {canonical_method(control) for control in CONTROL_NAMES}
        candidate_keys = [key for key in indexed if key not in control_keys]
        if len(candidate_keys) != 1:
            found = [display_names[key] for key in candidate_keys]
            raise ValueError(
                f"{csv_path}: expected exactly one selected candidate beside the four controls, "
                f"found {found}"
            )

        seed = discover_seed(csv_path, rows, fieldnames)
        candidate_key = candidate_keys[0]
        candidate_name = display_names[candidate_key]
        if candidate_key in candidate_names and candidate_names[candidate_key] != candidate_name:
            raise ValueError(
                f"Inconsistent display names for candidate key {candidate_key!r}: "
                f"{candidate_names[candidate_key]!r} and {candidate_name!r}"
            )
        candidate_names[candidate_key] = candidate_name
        candidate_values = read_values(
            csv_path, candidate_name, indexed[candidate_key], metric_columns
        )

        for comparator in CONTROL_NAMES:
            comparator_key = canonical_method(comparator)
            comparator_values = read_values(
                csv_path, comparator, indexed[comparator_key], metric_columns
            )
            raw_delta = {
                key: candidate_values[key] - comparator_values[key]
                for key in METRIC_BY_KEY
            }
            oriented_delta = {
                key: raw_delta[key] if METRIC_BY_KEY[key].higher_is_better else -raw_delta[key]
                for key in METRIC_BY_KEY
            }
            comparisons.append(
                {
                    "candidate_key": candidate_key,
                    "candidate": candidate_name,
                    "seed": seed,
                    "source": str(csv_path),
                    "comparator": comparator,
                    "comparator_values": comparator_values,
                    "candidate_values": candidate_values,
                    "raw_delta": raw_delta,
                    "oriented_delta": oriented_delta,
                }
            )

    if not comparisons:
        raise ValueError("No candidate comparisons were found")
    return comparisons, candidate_names


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def metric_statistics(records: Sequence[Mapping[str, object]], metric_key: str) -> dict[str, object]:
    comparator_values = [
        float(record["comparator_values"][metric_key]) for record in records  # type: ignore[index]
    ]
    candidate_values = [
        float(record["candidate_values"][metric_key]) for record in records  # type: ignore[index]
    ]
    raw_deltas = [
        float(record["raw_delta"][metric_key]) for record in records  # type: ignore[index]
    ]
    oriented_deltas = [
        float(record["oriented_delta"][metric_key]) for record in records  # type: ignore[index]
    ]
    wins = sum(delta > EPSILON for delta in oriented_deltas)
    ties = sum(abs(delta) <= EPSILON for delta in oriented_deltas)
    losses = sum(delta < -EPSILON for delta in oriented_deltas)
    return {
        "comparator_mean": statistics.fmean(comparator_values),
        "candidate_mean": statistics.fmean(candidate_values),
        "raw_delta_mean": statistics.fmean(raw_deltas),
        "raw_delta_std": sample_std(raw_deltas),
        "oriented_delta_mean": statistics.fmean(oriented_deltas),
        "oriented_delta_std": sample_std(oriented_deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(oriented_deltas),
    }


def summarize_candidates(
    comparisons: Sequence[Mapping[str, object]],
    candidate_names: Mapping[str, str],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in comparisons:
        grouped[str(record["candidate_key"])].append(record)

    summaries: list[dict[str, object]] = []
    for candidate_key in sorted(grouped):
        records = grouped[candidate_key]
        by_comparator: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for record in records:
            by_comparator[str(record["comparator"])].append(record)

        reference_records = by_comparator["Reference"]
        seeds = [str(record["seed"]) for record in reference_records]
        unique_seeds = set(seeds)
        completeness_reasons: list[str] = []
        if len(reference_records) != EXPECTED_SEEDS:
            completeness_reasons.append(
                f"requires exactly {EXPECTED_SEEDS} seed files, found {len(reference_records)}"
            )
        duplicate_seeds = sorted(
            {seed for seed in seeds if seeds.count(seed) > 1}, key=natural_sort_key
        )
        if duplicate_seeds:
            completeness_reasons.append(f"duplicate seed labels: {duplicate_seeds}")
        if len(unique_seeds) != EXPECTED_SEEDS:
            completeness_reasons.append(
                f"requires {EXPECTED_SEEDS} distinct seeds, found {len(unique_seeds)}"
            )
        complete = not completeness_reasons

        stats: dict[str, dict[str, dict[str, object]]] = {}
        for comparator in CONTROL_NAMES:
            comparator_records = by_comparator.get(comparator, [])
            if len(comparator_records) != len(reference_records):
                completeness_reasons.append(
                    f"{comparator} has {len(comparator_records)} pairs, expected {len(reference_records)}"
                )
                complete = False
            stats[comparator] = {
                metric.key: metric_statistics(comparator_records, metric.key)
                for metric in METRICS
            }

        criteria: dict[str, bool] = {}
        reasons = list(completeness_reasons)
        reference = stats["Reference"]
        for key in ("shape", "trend"):
            passed = int(reference[key]["wins"]) >= 4
            criteria[f"reference_{key}_wins_at_least_4_of_6"] = passed
            if not passed:
                reasons.append(
                    f"beats Reference on {METRIC_BY_KEY[key].label} in "
                    f"{reference[key]['wins']}/6 seeds; requires at least 4/6"
                )

        reference_positive_means = sum(
            float(reference[metric.key]["oriented_delta_mean"]) > EPSILON
            for metric in METRICS
        )
        criteria["reference_positive_mean_metrics_at_least_4_of_6"] = (
            reference_positive_means >= 4
        )
        if reference_positive_means < 4:
            reasons.append(
                f"positive mean delta vs Reference on {reference_positive_means}/6 metrics; "
                "requires at least 4/6"
            )

        primary_control_wins: dict[str, bool] = {}
        for comparator in ("Global", "Frozen"):
            passed = any(
                float(stats[comparator][key]["oriented_delta_mean"]) > EPSILON
                for key in ("shape", "trend")
            )
            primary_control_wins[comparator] = passed
            criteria[f"beats_{comparator.casefold()}_on_mean_shape_or_trend"] = passed
            if not passed:
                reasons.append(
                    f"does not beat {comparator} on mean Shape or mean Trend"
                )

        catastrophic_breaches: list[dict[str, object]] = []
        for comparator in CONTROL_NAMES:
            for metric_key, floor in CATASTROPHIC_FLOORS.items():
                value = float(stats[comparator][metric_key]["oriented_delta_mean"])
                if value < floor - EPSILON:
                    catastrophic_breaches.append(
                        {
                            "comparator": comparator,
                            "metric": metric_key,
                            "oriented_delta_mean": value,
                            "floor": floor,
                        }
                    )
                    reasons.append(
                        f"catastrophic guardrail: {METRIC_BY_KEY[metric_key].label} mean "
                        f"benefit delta vs {comparator} is {value:.6g}, below {floor:.6g}"
                    )
        criteria["no_catastrophic_guardrail_breach"] = not catastrophic_breaches

        criteria_pass = all(criteria.values())
        paper_grade_pass = complete and criteria_pass
        summaries.append(
            {
                "candidate_key": candidate_key,
                "candidate": candidate_names[candidate_key],
                "seed_count": len(reference_records),
                "distinct_seed_count": len(unique_seeds),
                "seeds": sorted(unique_seeds, key=natural_sort_key),
                "complete": complete,
                "paper_grade_pass": paper_grade_pass,
                "criteria": criteria,
                "reasons": reasons,
                "reference_positive_mean_metrics": reference_positive_means,
                "primary_control_wins": primary_control_wins,
                "catastrophic_breaches": catastrophic_breaches,
                "comparisons": stats,
            }
        )
    return summaries


def natural_sort_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


def rank_summaries(summaries: list[dict[str, object]]) -> None:
    ranked = sorted(
        summaries,
        key=lambda summary: (
            not bool(summary["paper_grade_pass"]),
            not bool(summary["complete"]),
            -int(summary["comparisons"]["Reference"]["shape"]["wins"]),  # type: ignore[index]
            -int(summary["comparisons"]["Reference"]["trend"]["wins"]),  # type: ignore[index]
            -int(summary["reference_positive_mean_metrics"]),
            str(summary["candidate"]).casefold(),
        ),
    )
    for rank, summary in enumerate(ranked, start=1):
        summary["rank"] = rank


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


def write_paired_csv(output_path: Path, comparisons: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "candidate",
        "seed",
        "comparator",
        "metric",
        "higher_is_better",
        "comparator_value",
        "candidate_value",
        "raw_delta",
        "oriented_delta",
        "outcome",
        "source",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(
            comparisons,
            key=lambda item: (
                str(item["candidate"]).casefold(),
                natural_sort_key(str(item["seed"])),
                CONTROL_NAMES.index(str(item["comparator"])),
            ),
        ):
            for metric in METRICS:
                delta = float(record["oriented_delta"][metric.key])  # type: ignore[index]
                outcome = "win" if delta > EPSILON else "loss" if delta < -EPSILON else "tie"
                writer.writerow(
                    {
                        "candidate": record["candidate"],
                        "seed": record["seed"],
                        "comparator": record["comparator"],
                        "metric": metric.key,
                        "higher_is_better": metric.higher_is_better,
                        "comparator_value": record["comparator_values"][metric.key],  # type: ignore[index]
                        "candidate_value": record["candidate_values"][metric.key],  # type: ignore[index]
                        "raw_delta": record["raw_delta"][metric.key],  # type: ignore[index]
                        "oriented_delta": delta,
                        "outcome": outcome,
                        "source": record["source"],
                    }
                )


def write_summary_csv(output_path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "rank",
        "candidate",
        "complete",
        "paper_grade_pass",
        "seed_count",
        "distinct_seed_count",
        "seeds",
        "comparator",
        "metric",
        "direction",
        "comparator_mean",
        "candidate_mean",
        "raw_delta_mean",
        "raw_delta_std",
        "oriented_delta_mean",
        "oriented_delta_std",
        "wins",
        "ties",
        "losses",
        "win_rate",
        "reasons",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda item: int(item["rank"])):
            for comparator in CONTROL_NAMES:
                for metric in METRICS:
                    values = summary["comparisons"][comparator][metric.key]  # type: ignore[index]
                    writer.writerow(
                        {
                            "rank": summary["rank"],
                            "candidate": summary["candidate"],
                            "complete": summary["complete"],
                            "paper_grade_pass": summary["paper_grade_pass"],
                            "seed_count": summary["seed_count"],
                            "distinct_seed_count": summary["distinct_seed_count"],
                            "seeds": ";".join(str(seed) for seed in summary["seeds"]),  # type: ignore[union-attr]
                            "comparator": comparator,
                            "metric": metric.key,
                            "direction": "higher" if metric.higher_is_better else "lower",
                            "comparator_mean": values["comparator_mean"],
                            "candidate_mean": values["candidate_mean"],
                            "raw_delta_mean": values["raw_delta_mean"],
                            "raw_delta_std": values["raw_delta_std"],
                            "oriented_delta_mean": values["oriented_delta_mean"],
                            "oriented_delta_std": values["oriented_delta_std"],
                            "wins": values["wins"],
                            "ties": values["ties"],
                            "losses": values["losses"],
                            "win_rate": values["win_rate"],
                            "reasons": "; ".join(str(reason) for reason in summary["reasons"]),  # type: ignore[arg-type]
                        }
                    )


def write_summary_markdown(
    output_path: Path,
    summaries: Sequence[Mapping[str, object]],
    csv_paths: Sequence[Path],
) -> None:
    passed = [str(summary["candidate"]) for summary in summaries if summary["paper_grade_pass"]]
    all_complete = all(bool(summary["complete"]) for summary in summaries)
    decision_rows = []
    for summary in sorted(summaries, key=lambda item: int(item["rank"])):
        reference = summary["comparisons"]["Reference"]  # type: ignore[index]
        decision_rows.append(
            (
                summary["rank"],
                summary["candidate"],
                f"{summary['distinct_seed_count']}/{EXPECTED_SEEDS}",
                "yes" if summary["complete"] else "no",
                "PASS" if summary["paper_grade_pass"] else "FAIL",
                f"{reference['shape']['wins']}/6",
                f"{reference['trend']['wins']}/6",
                f"{summary['reference_positive_mean_metrics']}/6",
                "yes" if summary["primary_control_wins"]["Global"] else "no",  # type: ignore[index]
                "yes" if summary["primary_control_wins"]["Frozen"] else "no",  # type: ignore[index]
            )
        )

    lines = [
        "# UAR confirmation summary",
        "",
        f"- Input files: **{len(csv_paths)}**",
        f"- Candidates: **{len(summaries)}**",
        f"- All candidates have six distinct seeds: **{'yes' if all_complete else 'no'}**",
        f"- Paper-grade pass: **{', '.join(passed) if passed else 'none'}**",
        "- Delta convention: candidate minus comparator, then sign-oriented so positive is better; "
        "for NRMSE, benefit delta is comparator minus candidate.",
        "",
        "## Decision",
        "",
        markdown_table(
            (
                "Rank",
                "Candidate",
                "Seeds",
                "Complete",
                "Decision",
                "Shape vs Ref",
                "Trend vs Ref",
                "Mean wins vs Ref",
                "Beats Global primary",
                "Beats Frozen primary",
            ),
            decision_rows,
        ),
        "",
    ]

    for summary in sorted(summaries, key=lambda item: int(item["rank"])):
        lines.extend([f"## {summary['candidate']}", ""])
        if summary["reasons"]:
            lines.append("Failure reasons:")
            lines.append("")
            lines.extend(f"- {reason}" for reason in summary["reasons"])
            lines.append("")
        metric_rows = []
        for comparator in CONTROL_NAMES:
            for metric in METRICS:
                values = summary["comparisons"][comparator][metric.key]  # type: ignore[index]
                metric_rows.append(
                    (
                        comparator,
                        metric.label,
                        format_number(values["comparator_mean"]),
                        format_number(values["candidate_mean"]),
                        format_number(values["oriented_delta_mean"]),
                        format_number(values["oriented_delta_std"]),
                        f"{values['wins']}/{summary['seed_count']}",
                    )
                )
        lines.extend(
            [
                markdown_table(
                    (
                        "Comparator",
                        "Metric",
                        "Comparator mean",
                        "Candidate mean",
                        "Benefit Δ mean",
                        "Benefit Δ std",
                        "Paired wins",
                    ),
                    metric_rows,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Paper-grade rule",
            "",
            f"- Exactly {EXPECTED_SEEDS} distinct, identifiable seeds per candidate; missing or duplicate seeds fail closed.",
            "- Shape and Trend each beat Reference in at least 4/6 paired seeds.",
            "- Candidate has a positive mean benefit delta vs Reference on at least 4/6 metrics.",
            "- Candidate beats both Global and Frozen on mean Shape or mean Trend (independently for each control).",
            "- No mean catastrophic breach vs Reference, Global, Frozen, or Backbone: "
            "Beta ≥ -0.10, R2 ≥ -0.10, NRMSE benefit ≥ -0.05.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def jsonable_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "rank": summary["rank"],
        "candidate": summary["candidate"],
        "seed_count": summary["seed_count"],
        "distinct_seed_count": summary["distinct_seed_count"],
        "seeds": summary["seeds"],
        "complete": summary["complete"],
        "paper_grade_pass": summary["paper_grade_pass"],
        "criteria": summary["criteria"],
        "reasons": summary["reasons"],
        "reference_positive_mean_metrics": summary["reference_positive_mean_metrics"],
        "primary_control_wins": summary["primary_control_wins"],
        "catastrophic_breaches": summary["catastrophic_breaches"],
        "comparisons": summary["comparisons"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate six-seed paired UAR confirmations against Reference, Global, "
            "Frozen and Backbone, then apply paper-grade pass rules."
        )
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        required=True,
        help="Glob for confirmation metrics.csv files; repeatable and supports **.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_paths = expand_inputs(args.input_glob)
    if not csv_paths:
        raise FileNotFoundError(f"No files matched --input-glob values: {args.input_glob}")

    comparisons, candidate_names = load_comparisons(csv_paths)
    summaries = summarize_candidates(comparisons, candidate_names)
    rank_summaries(summaries)
    ranked = sorted(summaries, key=lambda item: int(item["rank"]))

    args.output.mkdir(parents=True, exist_ok=True)
    write_paired_csv(args.output / "PAIRED_DELTAS.csv", comparisons)
    write_summary_csv(args.output / "SUMMARY.csv", ranked)
    write_summary_markdown(args.output / "SUMMARY.md", ranked, csv_paths)

    passed = [str(summary["candidate"]) for summary in ranked if summary["paper_grade_pass"]]
    decision = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": [str(path) for path in csv_paths],
        "expected_seeds_per_candidate": EXPECTED_SEEDS,
        "comparators": list(CONTROL_NAMES),
        "metrics": [metric.key for metric in METRICS],
        "delta_convention": {
            "raw_delta": "candidate - comparator",
            "oriented_delta": "positive is better; comparator - candidate for NRMSE",
        },
        "paper_grade_rule": {
            "reference_shape_paired_wins_minimum": 4,
            "reference_trend_paired_wins_minimum": 4,
            "reference_positive_mean_metrics_minimum": 4,
            "must_beat_mean_shape_or_trend_of_each": ["Global", "Frozen"],
            "catastrophic_floors_oriented": CATASTROPHIC_FLOORS,
            "catastrophic_floor_comparators": list(CONTROL_NAMES),
        },
        "all_candidates_complete": all(bool(summary["complete"]) for summary in ranked),
        "paper_grade_passed_candidates": passed,
        "candidates": [jsonable_summary(summary) for summary in ranked],
    }
    (args.output / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print((args.output / "SUMMARY.md").resolve())
    print(f"Paper-grade pass: {', '.join(passed) if passed else 'none'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
