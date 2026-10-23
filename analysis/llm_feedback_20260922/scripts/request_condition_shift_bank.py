#!/usr/bin/env python3
"""Build a synthetic-only language/composition shift benchmark for contracts.

This bank deliberately falls outside ``request_condition_local_baselines``'s
controlled template grammar.  It tests whether a compiler can preserve
semantics under held-out thresholds, paraphrases, aliases, conjunctions,
directionality, and explicit exceptions.  Every row originates from a marked
synthetic TabDiff preview; this script never reads real Adult rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from request_condition_common import evaluate_dsl, make_predicate, make_repair


ADVANCED = ["Doctorate", "Prof-school", "Masters"]


def contract(
    request_id: str,
    family: str,
    shift: str,
    text: str,
    antecedent: list[dict],
    requirement: list[dict],
    repair: dict,
    exceptions: list[dict] | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "family": family,
        "shift": shift,
        "text": text,
        "oracle_dsl": {
            "antecedent_all": antecedent,
            "requirement_all": requirement,
            "exception_all": exceptions or [],
            "repair": repair,
        },
    }


def request_catalog() -> list[dict]:
    return [
        contract(
            "advanced_age_22", "advanced_age", "heldout_threshold_paraphrase",
            "Treat Masters, professional-school, and doctorate qualifications as advanced. "
            "A row with one of those qualifications is admissible only after age twenty-two; "
            "other education values are outside this policy. If short, increase age to 22.",
            [make_predicate("education", "in", string_list=ADVANCED)],
            [make_predicate("age", "gte", number_value=22)],
            make_repair("age", number_value=22),
        ),
        contract(
            "advanced_age_34", "advanced_age", "heldout_threshold_reordered",
            "Minimum age: 34. This floor applies precisely when education carries any of "
            "Doctorate, Prof-school, or Masters, and otherwise the request does not apply. "
            "Correct a breach by replacing age with 34.",
            [make_predicate("education", "in", string_list=ADVANCED)],
            [make_predicate("age", "gte", number_value=34)],
            make_repair("age", number_value=34),
        ),
        contract(
            "private_weekly_28", "private_hours", "column_alias_paraphrase",
            "For employees in the private sector (workclass value Private), weekly workload—"
            "the schema field measuring hours in a week—has a floor of 28. Raise that field "
            "to 28 when it falls short; non-Private rows are out of scope.",
            [make_predicate("workclass", "eq", string_value="Private")],
            [make_predicate("hours.per.week", "gte", number_value=28)],
            make_repair("hours.per.week", number_value=28),
        ),
        contract(
            "private_weekly_46", "private_hours", "heldout_threshold_logic_form",
            "A Private workclass entry with fewer than 46 in hours.per.week breaches this "
            "request; every other workclass is not governed by it. For a breach, replace "
            "hours.per.week by 46.",
            [make_predicate("workclass", "eq", string_value="Private")],
            [make_predicate("hours.per.week", "gte", number_value=46)],
            make_repair("hours.per.week", number_value=46),
        ),
        contract(
            "advanced_private_weekly_32", "advanced_private_hours", "antecedent_composition",
            "Activate this contract only when both are true: education belongs to Masters, "
            "Prof-school, or Doctorate; and workclass is Private. In that intersection, "
            "hours.per.week must be at least 32. Set it to 32 for a true violation.",
            [
                make_predicate("education", "in", string_list=ADVANCED),
                make_predicate("workclass", "eq", string_value="Private"),
            ],
            [make_predicate("hours.per.week", "gte", number_value=32)],
            make_repair("hours.per.week", number_value=32),
        ),
        contract(
            "advanced_private_weekly_48", "advanced_private_hours", "composition_negated_scope",
            "Unless the record is simultaneously Private in workclass and advanced in "
            "education (Doctorate, Masters, or Prof-school), ignore this request. Records in "
            "that joint scope need 48 or more hours.per.week; lift a deficient value to 48.",
            [
                make_predicate("workclass", "eq", string_value="Private"),
                make_predicate("education", "in", string_list=ADVANCED),
            ],
            [make_predicate("hours.per.week", "gte", number_value=48)],
            make_repair("hours.per.week", number_value=48),
        ),
        contract(
            "spouse_exception_rephrased", "categorical_exception", "exception_paraphrase",
            "Normally a relationship label of Husband or Wife entails marital.status "
            "Married-civ-spouse. Married-AF-spouse is a permitted carve-out and must remain "
            "untouched. For any other mismatch in scope, use Married-civ-spouse.",
            [make_predicate("relationship", "in", string_list=["Husband", "Wife"])],
            [make_predicate("marital.status", "eq", string_value="Married-civ-spouse")],
            make_repair("marital.status", string_value="Married-civ-spouse"),
            [make_predicate("marital.status", "eq", string_value="Married-AF-spouse")],
        ),
        contract(
            "missing_exception_rephrased", "missingness_exception", "exception_directionality",
            "A question mark in occupation activates the policy. It then calls for a question "
            "mark in workclass too, except that workclass Never-worked is explicitly allowed. "
            "Only a non-exempt mismatch should have workclass changed to '?'.",
            [make_predicate("occupation", "eq", string_value="?")],
            [make_predicate("workclass", "eq", string_value="?")],
            make_repair("workclass", string_value="?"),
            [make_predicate("workclass", "eq", string_value="Never-worked")],
        ),
        contract(
            "positive_gain_zero_loss", "capital_direction", "operator_shift_lte",
            "Whenever capital.gain is strictly positive, capital.loss may not exceed zero. "
            "A row with no positive gain is outside the rule. Set capital.loss to 0 if the "
            "active rule is violated.",
            [make_predicate("capital.gain", "gt", number_value=0)],
            [make_predicate("capital.loss", "lte", number_value=0)],
            make_repair("capital.loss", number_value=0),
        ),
        contract(
            "foreign_gain_cap", "country_gain_cap", "operator_shift_neq_lte",
            "For a native.country other than United-States, capital.gain must be no greater "
            "than 5000. United-States records are not applicable. Clamp a foreign record that "
            "exceeds the limit to 5000.",
            [make_predicate("native.country", "neq", string_value="United-States")],
            [make_predicate("capital.gain", "lte", number_value=5000)],
            make_repair("capital.gain", number_value=5000),
        ),
    ]


def set_string(row: dict, column: str, value: str) -> None:
    row[column] = f" {value}"


def add_case(
    cases: list[dict], base: dict, request: dict, row: dict, row_id: str,
    design: str, pair_id: str | None = None, expected: str | None = None,
) -> None:
    status = evaluate_dsl(row, request["oracle_dsl"])
    if expected is not None and status != expected:
        raise AssertionError((request["request_id"], expected, status, row))
    cases.append(
        {
            "case_id": f"shiftcase_{len(cases):04d}",
            "row_id": row_id,
            "pair_id": pair_id,
            "trajectory_id": base["trajectory_id"],
            "t": float(base["t"]),
            "dataset": "adult",
            "record": row,
            "request_id": request["request_id"],
            "request_family": request["family"],
            "request_shift": request["shift"],
            "design": design,
            "gold_status": status,
            "gold_action": "repair_candidate" if status == "violated" else "no_action",
            "privacy": "synthetic_preview_only",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    source = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    clean = [case for case in source if case.get("control_type") == "paired_clean"]
    if not clean or any(case.get("privacy") != "synthetic_preview_only" for case in clean):
        raise RuntimeError("source must contain marked synthetic paired-clean previews only")
    by_time: dict[float, list[dict]] = {}
    for case in clean:
        by_time.setdefault(float(case["t"]), []).append(case)

    requests = request_catalog()
    lookup = {item["request_id"]: item for item in requests}
    cases: list[dict] = []
    for time_index, (_, pool) in enumerate(sorted(by_time.items(), reverse=True)):
        cursor = 0

        def base() -> dict:
            nonlocal cursor
            item = pool[cursor % len(pool)]
            cursor += 1
            return item

        for replicate in range(args.replicates):
            # Three same-row/different-request pairs.  Row plausibility alone
            # cannot determine which member should be repaired.
            item = base(); row = dict(item["record"])
            set_string(row, "education", "Masters"); row["age"] = 28.0
            pair = f"age_shift_t{time_index}_r{replicate}"
            for request_id in ("advanced_age_22", "advanced_age_34"):
                add_case(cases, item, lookup[request_id], dict(row), pair, "same_row_heldout_threshold", pair)

            item = base(); row = dict(item["record"])
            set_string(row, "workclass", "Private"); row["hours.per.week"] = 36.0
            pair = f"hours_shift_t{time_index}_r{replicate}"
            for request_id in ("private_weekly_28", "private_weekly_46"):
                add_case(cases, item, lookup[request_id], dict(row), pair, "same_row_alias_threshold", pair)

            item = base(); row = dict(item["record"])
            set_string(row, "education", "Doctorate"); set_string(row, "workclass", "Private")
            row["hours.per.week"] = 40.0
            pair = f"composition_shift_t{time_index}_r{replicate}"
            for request_id in ("advanced_private_weekly_32", "advanced_private_weekly_48"):
                add_case(cases, item, lookup[request_id], dict(row), pair, "same_row_composed_threshold", pair)

            # Conjunction controls: each violates the consequent numerically,
            # but exactly one antecedent is false, so both are not applicable.
            for request_id, education, workclass in (
                ("advanced_private_weekly_32", "Bachelors", "Private"),
                ("advanced_private_weekly_48", "Masters", "Self-emp-not-inc"),
            ):
                item = base(); row = dict(item["record"])
                set_string(row, "education", education); set_string(row, "workclass", workclass)
                row["hours.per.week"] = 10.0
                add_case(cases, item, lookup[request_id], row, f"comp_na_t{time_index}_r{replicate}_{request_id}",
                         "composition_not_applicable", expected="not_applicable")

            for expected, relationship, marital in (
                ("excepted", "Husband", "Married-AF-spouse"),
                ("satisfied", "Wife", "Married-civ-spouse"),
                ("violated", "Husband", "Never-married"),
                ("not_applicable", "Own-child", "Never-married"),
            ):
                item = base(); row = dict(item["record"])
                set_string(row, "relationship", relationship); set_string(row, "marital.status", marital)
                add_case(cases, item, lookup["spouse_exception_rephrased"], row,
                         f"spouse_{expected}_t{time_index}_r{replicate}", f"exception_{expected}", expected=expected)

            for expected, occupation, workclass in (
                ("excepted", "?", "Never-worked"),
                ("satisfied", "?", "?"),
                ("violated", "?", "Private"),
                ("not_applicable", "Craft-repair", "Private"),
            ):
                item = base(); row = dict(item["record"])
                set_string(row, "occupation", occupation); set_string(row, "workclass", workclass)
                add_case(cases, item, lookup["missing_exception_rephrased"], row,
                         f"missing_{expected}_t{time_index}_r{replicate}", f"exception_{expected}", expected=expected)

            for expected, gain, loss in (
                ("satisfied", 1000.0, 0.0),
                ("violated", 1000.0, 250.0),
                ("not_applicable", 0.0, 250.0),
            ):
                item = base(); row = dict(item["record"])
                row["capital.gain"] = gain; row["capital.loss"] = loss
                add_case(cases, item, lookup["positive_gain_zero_loss"], row,
                         f"capital_{expected}_t{time_index}_r{replicate}", f"operator_{expected}", expected=expected)

            for expected, country, gain in (
                ("satisfied", "Mexico", 4000.0),
                ("violated", "Canada", 6000.0),
                ("not_applicable", "United-States", 9000.0),
            ):
                item = base(); row = dict(item["record"])
                set_string(row, "native.country", country); row["capital.gain"] = gain
                add_case(cases, item, lookup["foreign_gain_cap"], row,
                         f"country_{expected}_t{time_index}_r{replicate}", f"operator_{expected}", expected=expected)

    with (args.output / "requests.jsonl").open("w") as handle:
        for request in requests:
            handle.write(json.dumps(request) + "\n")
    with (args.output / "cases.jsonl").open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")
    metadata = {
        "status": "executed",
        "cases": len(cases),
        "requests": len(requests),
        "times": sorted(by_time, reverse=True),
        "status_counts": dict(Counter(case["gold_status"] for case in cases)),
        "design_counts": dict(Counter(case["design"] for case in cases)),
        "swap_pairs": len({case["pair_id"] for case in cases if case["pair_id"]}),
        "source_policy": "Only paired-clean synthetic TabDiff previews; no real rows.",
        "leakage_policy": "Gold, DSL, pair/design labels stay local; API runner uses opaque shuffled IDs.",
        "generalization_scope": [
            "heldout thresholds", "paraphrase/reordering", "column alias",
            "conjunctive antecedent", "explicit exceptions", "neq/gt/lte operators",
        ],
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
