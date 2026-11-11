#!/usr/bin/env python3
"""Train-defined, heldout-answered query fidelity for synthetic tables.

The workload is constructed only from the training split.  Answers are then
compared between the untouched heldout split and each synthetic table.  Query
column pairs already used by the supplied factor graph are excluded, making
this an external evaluator rather than another view of the optimized graph
objective.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_sample(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("sample must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("sample label is empty")
    return label, Path(path)


def graph_pairs(path: Path | None) -> set[frozenset[str]]:
    if path is None:
        return set()
    edges = json.loads(path.read_text())
    pairs: set[frozenset[str]] = set()
    for edge in edges:
        columns: set[str] = set(edge.get("columns", []))
        for section in ("applicability", "requirements"):
            for atom in edge.get(section, []):
                left = atom.get("left_column")
                right = atom.get("right_column") if atom.get("right_kind") == "column" else None
                if left:
                    columns.add(left)
                if right:
                    columns.add(right)
        ordered = sorted(columns)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                pairs.add(frozenset((left, right)))
    return pairs


def clean_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<NA>").str.strip()


def condition_mask(frame: pd.DataFrame, conditions: list[dict[str, Any]]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for condition in conditions:
        column = condition["column"]
        if column not in frame:
            return pd.Series(False, index=frame.index)
        if condition["kind"] == "eq":
            mask &= clean_category(frame[column]) == str(condition["value"])
        elif condition["kind"] == "interval":
            values = pd.to_numeric(frame[column], errors="coerce")
            lower = values >= float(condition["lower"])
            upper = values <= float(condition["upper"]) if condition.get("closed_upper") else values < float(condition["upper"])
            mask &= lower & upper
        else:
            raise ValueError(f"unknown condition kind: {condition['kind']}")
    return mask.fillna(False)


def query_key(aggregate: str, conditions: list[dict[str, Any]], measure: str | None) -> str:
    parts = [aggregate]
    for condition in conditions:
        if condition["kind"] == "eq":
            parts.append(f"{condition['column']}={condition['value']}")
        else:
            parts.append(
                f"{condition['column']}[{condition['lower']:.6g},{condition['upper']:.6g}"
                f"{']' if condition.get('closed_upper') else ')'}"
            )
    if measure:
        parts.append(measure)
    return "|".join(parts)


def add_query(
    workload: list[dict[str, Any]], seen: set[str], aggregate: str,
    conditions: list[dict[str, Any]], train_support: float, measure: str | None = None,
) -> None:
    key = query_key(aggregate, conditions, measure)
    if key in seen:
        return
    seen.add(key)
    workload.append({
        "query_id": f"q{len(workload):04d}",
        "aggregate": aggregate,
        "conditions": conditions,
        "measure": measure,
        "train_support": float(train_support),
        "rare": bool(train_support < 0.05),
        "key": key,
    })


def build_workload(
    train: pd.DataFrame, info: dict[str, Any], excluded_pairs: set[frozenset[str]],
    max_queries: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    names = info["column_names"]
    numerical = [names[i] for i in info.get("num_col_idx", []) if names[i] in train]
    categorical = [names[i] for i in info.get("cat_col_idx", []) if names[i] in train]
    target = {names[i] for i in info.get("target_col_idx", [])}
    # Prefer useful non-degenerate measures, while retaining a numerical target
    # when the task is regression.
    scales: dict[str, float] = {}
    for column in numerical:
        values = pd.to_numeric(train[column], errors="coerce")
        q25, q75 = values.quantile([0.25, 0.75])
        scales[column] = float(max(q75 - q25, values.std(), 1e-8))
    measures = sorted(numerical, key=lambda c: (c not in target, -scales[c]))[:5]

    workload: list[dict[str, Any]] = []
    seen: set[str] = set()
    n = max(len(train), 1)

    # One-dimensional categorical GROUP BY cells: common plus rare groups.
    cat_columns = []
    for column in categorical:
        values = clean_category(train[column])
        cardinality = values.nunique(dropna=False)
        if 2 <= cardinality <= 32:
            entropy = float(-(values.value_counts(normalize=True) * np.log(values.value_counts(normalize=True))).sum())
            cat_columns.append((entropy, column))
    cat_columns = [column for _, column in sorted(cat_columns, reverse=True)[:8]]
    base_predicates: list[tuple[list[dict[str, Any]], float]] = []
    for column in cat_columns:
        frequencies = clean_category(train[column]).value_counts(normalize=True)
        chosen = list(frequencies.head(2).index)
        rare = frequencies[(frequencies >= 0.005) & (frequencies < 0.05)].sort_values().head(2)
        chosen.extend([value for value in rare.index if value not in chosen])
        for value in chosen:
            support = float(frequencies[value])
            conditions = [{"kind": "eq", "column": column, "value": str(value)}]
            base_predicates.append((conditions, support))
            add_query(workload, seen, "count", conditions, support)

    # Numerical tails and central ranges, all thresholds fitted on train.
    numeric_columns = sorted(numerical, key=lambda c: -scales[c])[:6]
    for column in numeric_columns:
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        if values.nunique() < 8:
            continue
        quantiles = values.quantile([0.0, 0.1, 0.25, 0.75, 0.9, 1.0]).to_dict()
        intervals = [(0.0, 0.1), (0.1, 0.25), (0.75, 0.9), (0.9, 1.0)]
        for lower_q, upper_q in intervals:
            lower, upper = float(quantiles[lower_q]), float(quantiles[upper_q])
            if not math.isfinite(lower + upper) or upper <= lower:
                continue
            conditions = [{
                "kind": "interval", "column": column, "lower": lower,
                "upper": upper, "closed_upper": upper_q == 1.0,
            }]
            support = float(condition_mask(train, conditions).mean())
            base_predicates.append((conditions, support))
            add_query(workload, seen, "count", conditions, support)

    # Two-column GROUP BY cells.  Exclude factor-graph pairs by construction.
    for left_index, left in enumerate(cat_columns):
        for right in cat_columns[left_index + 1 :]:
            if frozenset((left, right)) in excluded_pairs:
                continue
            pair = pd.DataFrame({left: clean_category(train[left]), right: clean_category(train[right])})
            frequencies = pair.value_counts(normalize=True)
            common = frequencies[(frequencies >= 0.05) & (frequencies <= 0.5)].head(1)
            rare = frequencies[(frequencies >= 0.005) & (frequencies < 0.05)].sort_values().head(1)
            for values, support in pd.concat([common, rare]).items():
                conditions = [
                    {"kind": "eq", "column": left, "value": str(values[0])},
                    {"kind": "eq", "column": right, "value": str(values[1])},
                ]
                base_predicates.append((conditions, float(support)))
                add_query(workload, seen, "count", conditions, float(support))
            if len(workload) >= max_queries // 2:
                break
        if len(workload) >= max_queries // 2:
            break

    # Conditional AVG and normalized SUM queries.  The conditioning pair must
    # remain outside the optimized graph, but individual columns may overlap.
    for conditions, support in base_predicates:
        condition_columns = {condition["column"] for condition in conditions}
        for measure in measures[:2]:
            if measure in condition_columns:
                continue
            if any(frozenset((measure, column)) in excluded_pairs for column in condition_columns):
                continue
            add_query(workload, seen, "avg", conditions, support, measure)
            add_query(workload, seen, "sum", conditions, support, measure)
            break
        if len(workload) >= max_queries:
            break

    return workload[:max_queries], scales


def answer(frame: pd.DataFrame, query: dict[str, Any]) -> tuple[float, int]:
    mask = condition_mask(frame, query["conditions"])
    support = int(mask.sum())
    aggregate = query["aggregate"]
    if aggregate == "count":
        return float(mask.mean()), support
    values = pd.to_numeric(frame.loc[mask, query["measure"]], errors="coerce").dropna()
    if aggregate == "avg":
        return (float(values.mean()) if len(values) else float("nan")), support
    if aggregate == "sum":
        return float(values.sum() / max(len(frame), 1)), support
    raise ValueError(aggregate)


def evaluate_sample(
    label: str, synthetic: pd.DataFrame, heldout: pd.DataFrame,
    workload: list[dict[str, Any]], scales: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for query in workload:
        real, real_rows = answer(heldout, query)
        synth, synth_rows = answer(synthetic, query)
        aggregate = query["aggregate"]
        if aggregate == "count":
            epsilon = 0.5 / max(len(heldout), len(synthetic), 1)
            qerror = max((real + epsilon) / (synth + epsilon), (synth + epsilon) / (real + epsilon))
            normalized_error = abs(synth - real)
        else:
            qerror = float("nan")
            scale = scales.get(query["measure"], 1.0)
            normalized_error = abs(synth - real) / scale if math.isfinite(real) and math.isfinite(synth) else float("nan")
        smape = (
            2 * abs(synth - real) / max(abs(synth) + abs(real), 1e-12)
            if math.isfinite(real) and math.isfinite(synth) else float("nan")
        )
        # TabQueryBench-compatible semantic score.  Count queries return a
        # bounded rate, so their primary score is 1-MAE.  AVG/SUM answers are
        # numeric magnitudes and use normalized sMAPE similarity
        # 1-|r-s|/(|r|+|s|).  Workload construction remains ours: train-only
        # and graph-pair excluded.  Keeping construction and scoring separate
        # makes the evaluator independent without inventing a paper-specific
        # fidelity scale.
        if not (math.isfinite(real) and math.isfinite(synth)):
            semantic_score = float("nan")
            semantic_scorer = "invalid"
        elif aggregate == "count":
            semantic_score = float(1.0 - min(abs(synth - real), 1.0))
            semantic_scorer = "rate_1_minus_mae"
        elif real == 0.0 and synth == 0.0:
            semantic_score = 1.0
            semantic_scorer = "numeric_smape_similarity"
        else:
            semantic_score = float(
                1.0 - abs(synth - real) / max(abs(synth) + abs(real), 1e-12)
            )
            semantic_scorer = "numeric_smape_similarity"
        rows.append({
            "method": label, "query_id": query["query_id"], "aggregate": aggregate,
            "key": query["key"], "measure": query["measure"],
            "train_support": query["train_support"], "rare": query["rare"],
            "real_answer": real, "synthetic_answer": synth,
            "real_rows": real_rows, "synthetic_rows": synth_rows,
            "normalized_error": normalized_error, "smape": smape, "qerror": qerror,
            "semantic_scorer": semantic_scorer,
            "query_centric_score": semantic_score,
        })
    frame = pd.DataFrame(rows)
    valid = frame["normalized_error"].dropna()
    count = frame[frame["aggregate"] == "count"]
    rare = frame[frame["rare"]]
    summary = {
        "method": label,
        "queries": int(len(frame)),
        "valid_queries": int(len(valid)),
        "query_fidelity_percent": float(100 * (1 - np.clip(valid, 0, 1).mean())),
        "query_centric_fidelity": float(frame.query_centric_score.mean()),
        "query_centric_fidelity_percent": float(100 * frame.query_centric_score.mean()),
        "query_centric_valid": int(frame.query_centric_score.notna().sum()),
        "mean_normalized_error": float(valid.mean()),
        "p90_normalized_error": float(valid.quantile(0.90)),
        "mean_smape": float(frame.smape.mean()),
        "count_mae_pp": float(100 * count.normalized_error.mean()),
        "count_qerror_median": float(count.qerror.median()),
        "count_qerror_p90": float(count.qerror.quantile(0.90)),
        "rare_mean_normalized_error": float(rare.normalized_error.mean()),
        "rare_p90_normalized_error": float(rare.normalized_error.quantile(0.90)),
        "rare_support_recall": float(((rare.synthetic_rows > 0) & (rare.real_rows > 0)).mean()),
        "rare_query_centric_fidelity": float(rare.query_centric_score.mean()),
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--sample", action="append", type=parse_sample, required=True)
    parser.add_argument("--max-queries", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    del args.seed  # Workload ordering is deterministic; retained for provenance compatibility.
    args.output.mkdir(parents=True, exist_ok=False)

    data = args.repo / "data" / args.dataset
    train = pd.read_csv(data / "train.csv")
    heldout = pd.read_csv(data / "test.csv")
    info = json.loads((data / "info.json").read_text())
    excluded = graph_pairs(args.graph)
    workload, scales = build_workload(train, info, excluded, args.max_queries)
    if len(workload) < 20:
        raise RuntimeError(f"workload too small: {len(workload)}")
    (args.output / "workload.json").write_text(json.dumps({
        "dataset": args.dataset,
        "construction_split": "train_only",
        "answer_reference_split": "untouched_test",
        "graph_pairs_excluded": len(excluded),
        "queries": workload,
    }, indent=2))

    details = []
    summaries = []
    for label, path in args.sample:
        synthetic = pd.read_csv(path)
        missing = sorted(set(train.columns) - set(synthetic.columns))
        if missing:
            raise RuntimeError(f"{label} missing columns: {missing}")
        detail, summary = evaluate_sample(label, synthetic, heldout, workload, scales)
        details.append(detail)
        summaries.append(summary)
    detailed = pd.concat(details, ignore_index=True)
    aggregate = pd.DataFrame(summaries)
    detailed.to_csv(args.output / "per_query.csv", index=False)
    aggregate.to_csv(args.output / "aggregate.csv", index=False)
    lines = [
        f"# Query-centric fidelity: {args.dataset}", "",
        "Workload predicates and aggregation targets are selected from train only; answers use the untouched test split.",
        "All column pairs present in the optimized factor graph are excluded.", "",
        "```csv", aggregate.to_csv(index=False).strip(), "```", "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
