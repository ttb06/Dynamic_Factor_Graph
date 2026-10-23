#!/usr/bin/env python3
"""Post-hoc audit of abstention and typed execution gates for LLM critiques."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

import llm_feedback_audit as audit


NO_OP_PHRASES = (
    "no change",
    "none needed",
    "no correction",
    "no conflict",
    "already valid",
    "already satisfies",
)


def is_no_op(item: dict) -> bool:
    text = str(item.get("suggested_change", "")).strip().lower()
    return not text or any(phrase in text for phrase in NO_OP_PHRASES)


def antecedent_allows(rule: str, record: dict) -> bool:
    clean = {key: str(value).strip() if isinstance(value, str) else value for key, value in record.items()}
    relationship = clean.get("relationship")
    if rule == "husband_male":
        return relationship == "Husband"
    if rule == "wife_female":
        return relationship == "Wife"
    if rule == "spouse_marital":
        return relationship in {"Husband", "Wife"}
    if rule == "capital_gain_nonnegative":
        return float(clean.get("capital.gain", 0)) < 0
    if rule == "capital_loss_nonnegative":
        return float(clean.get("capital.loss", 0)) < 0
    if rule == "hours_positive":
        return float(clean.get("hours.per.week", 1)) < 1
    # The education mapping is not inferable from the antecedent alone.
    return True


def gated_predictions(critique: dict, record: dict, gate: str) -> set[str]:
    result = set()
    for item in critique.get("critiques", []):
        if gate in {"no_op_filter", "typed_contract"} and is_no_op(item):
            continue
        item_predictions, _, _ = audit.canonical_predictions({"critiques": [item]})
        if gate == "typed_contract":
            item_predictions = {rule for rule in item_predictions if antecedent_allows(rule, record)}
        result.update(item_predictions)
    return result


def scores(predictions: list[set[str]], golds: list[set[str]]) -> dict[str, float | int]:
    tp = sum(len(pred & gold) for pred, gold in zip(predictions, golds))
    fp = sum(len(pred - gold) for pred, gold in zip(predictions, golds))
    fn = sum(len(gold - pred) for pred, gold in zip(predictions, golds))
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--schema-only", type=Path, required=True)
    parser.add_argument("--static-rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    cases = {
        item["case_id"]: item
        for item in (json.loads(line) for line in args.cases.read_text().splitlines() if line.strip())
    }
    inputs = {
        "schema_only": args.schema_only,
        "static_validated_relations": args.static_rules,
    }
    detail_rows = []
    metric_rows = []
    for condition, path in inputs.items():
        raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_case = defaultdict(list)
        for response in raw:
            case = cases[response["case_id"]]
            gold = set(case.get("verifier_violations", []))
            for gate in ("raw", "no_op_filter", "typed_contract"):
                pred = gated_predictions(response["critique"], case["record"], gate)
                detail_rows.append(
                    {
                        "condition": condition,
                        "case_id": response["case_id"],
                        "t": case["t"],
                        "repeat": response["repeat"],
                        "gate": gate,
                        "gold": "|".join(sorted(gold)),
                        "predicted": "|".join(sorted(pred)),
                    }
                )
                by_case[(response["case_id"], gate)].append(pred)

        details = [row for row in detail_rows if row["condition"] == condition]
        for gate in ("raw", "no_op_filter", "typed_contract"):
            selected = [row for row in details if row["gate"] == gate]
            predictions = [set(filter(None, row["predicted"].split("|"))) for row in selected]
            golds = [set(filter(None, row["gold"].split("|"))) for row in selected]
            metric_rows.append(
                {
                    "condition": condition,
                    "gate": gate,
                    "aggregation": "per_call",
                    "units": len(selected),
                    "accepted_relation_count": sum(map(len, predictions)),
                    **scores(predictions, golds),
                }
            )

            case_predictions, case_golds = [], []
            for case_id in sorted({key[0] for key in by_case if key[1] == gate}):
                repeats = by_case[(case_id, gate)]
                consensus = set.intersection(*repeats) if repeats else set()
                case_predictions.append(consensus)
                case_golds.append(set(cases[case_id].get("verifier_violations", [])))
            metric_rows.append(
                {
                    "condition": condition,
                    "gate": gate,
                    "aggregation": "repeat_intersection",
                    "units": len(case_predictions),
                    "accepted_relation_count": sum(map(len, case_predictions)),
                    **scores(case_predictions, case_golds),
                }
            )

    details_df = pd.DataFrame(detail_rows)
    metrics_df = pd.DataFrame(metric_rows)
    details_df.to_csv(args.output / "critique_gate_details.csv", index=False)
    metrics_df.to_csv(args.output / "critique_gate_metrics.csv", index=False)
    summary = {
        "status": "executed",
        "scope": "post-hoc; no new API calls",
        "no_op_filter": "Rejects suggested_change containing explicit no-action language.",
        "typed_contract": "Adds executable antecedent/range guards; education remains unguarded because mapping knowledge is required.",
        "caveat": "The typed contract partially overlaps a rule verifier and is evaluated only on 9 cases x 2 repeats per condition.",
        "metrics": metrics_df.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
