#!/usr/bin/env python3
"""Build a synthetic-only request-conditioned constraint benchmark.

Rows are derived exclusively from TabDiff synthetic previews.  The same row is
paired with two different threshold requests, and exception / not-applicable
controls make a binary "looks unusual" heuristic insufficient.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from request_condition_common import evaluate_dsl, make_predicate, make_repair


def request_catalog() -> list[dict]:
    degree_values = ["Doctorate", "Prof-school", "Masters"]
    return [
        {
            "request_id": "degree_age_18",
            "family": "degree_age_threshold",
            "text": (
                "For this generation request, when education is Doctorate, Prof-school, or Masters, "
                "age must be at least 18 years. There are no exceptions. If violated, set age to 18."
            ),
            "oracle_dsl": {
                "antecedent_all": [make_predicate("education", "in", string_list=degree_values)],
                "requirement_all": [make_predicate("age", "gte", number_value=18)],
                "exception_all": [],
                "repair": make_repair("age", number_value=18),
            },
        },
        {
            "request_id": "degree_age_30",
            "family": "degree_age_threshold",
            "text": (
                "For this generation request, when education is Doctorate, Prof-school, or Masters, "
                "age must be at least 30 years. There are no exceptions. If violated, set age to 30."
            ),
            "oracle_dsl": {
                "antecedent_all": [make_predicate("education", "in", string_list=degree_values)],
                "requirement_all": [make_predicate("age", "gte", number_value=30)],
                "exception_all": [],
                "repair": make_repair("age", number_value=30),
            },
        },
        {
            "request_id": "private_hours_25",
            "family": "private_hours_threshold",
            "text": (
                "For this generation request, if workclass is Private, hours.per.week must be at least 25. "
                "There are no exceptions. If violated, set hours.per.week to 25."
            ),
            "oracle_dsl": {
                "antecedent_all": [make_predicate("workclass", "eq", string_value="Private")],
                "requirement_all": [make_predicate("hours.per.week", "gte", number_value=25)],
                "exception_all": [],
                "repair": make_repair("hours.per.week", number_value=25),
            },
        },
        {
            "request_id": "private_hours_45",
            "family": "private_hours_threshold",
            "text": (
                "For this generation request, if workclass is Private, hours.per.week must be at least 45. "
                "There are no exceptions. If violated, set hours.per.week to 45."
            ),
            "oracle_dsl": {
                "antecedent_all": [make_predicate("workclass", "eq", string_value="Private")],
                "requirement_all": [make_predicate("hours.per.week", "gte", number_value=45)],
                "exception_all": [],
                "repair": make_repair("hours.per.week", number_value=45),
            },
        },
        {
            "request_id": "spouse_marital_with_af_exception",
            "family": "categorical_exception",
            "text": (
                "If relationship is Husband or Wife, marital.status must be Married-civ-spouse. "
                "Exception: Married-AF-spouse is explicitly allowed and must not be changed. "
                "For a true violation, set marital.status to Married-civ-spouse."
            ),
            "oracle_dsl": {
                "antecedent_all": [make_predicate("relationship", "in", string_list=["Husband", "Wife"])],
                "requirement_all": [make_predicate("marital.status", "eq", string_value="Married-civ-spouse")],
                "exception_all": [make_predicate("marital.status", "eq", string_value="Married-AF-spouse")],
                "repair": make_repair("marital.status", string_value="Married-civ-spouse"),
            },
        },
        {
            "request_id": "occupation_missing_with_never_worked_exception",
            "family": "missingness_exception",
            "text": (
                "If occupation is '?', workclass must also be '?'. Exception: the combination "
                "workclass=Never-worked and occupation='?' is explicitly allowed and must not be changed. "
                "For a true violation, set workclass to '?'."
            ),
            "oracle_dsl": {
                "antecedent_all": [make_predicate("occupation", "eq", string_value="?")],
                "requirement_all": [make_predicate("workclass", "eq", string_value="?")],
                "exception_all": [make_predicate("workclass", "eq", string_value="Never-worked")],
                "repair": make_repair("workclass", string_value="?"),
            },
        },
    ]


def set_string(record: dict, column: str, value: str) -> None:
    # Preserve the whitespace convention used by the Adult decoder.
    record[column] = f" {value}"


def add_case(cases: list[dict], base: dict, request: dict, row: dict, row_id: str, pair_id: str | None, design: str) -> None:
    status = evaluate_dsl(row, request["oracle_dsl"])
    cases.append(
        {
            "case_id": f"reqcase_{len(cases):04d}",
            "row_id": row_id,
            "pair_id": pair_id,
            "trajectory_id": base["trajectory_id"],
            "t": float(base["t"]),
            "dataset": "adult",
            "record": row,
            "request_id": request["request_id"],
            "request_family": request["family"],
            "request_text": request["text"],
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
    parser.add_argument("--threshold-replicates", type=int, default=3)
    parser.add_argument("--control-replicates", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    source = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    clean = [case for case in source if case.get("control_type") == "paired_clean"]
    if not clean or any(case.get("privacy") != "synthetic_preview_only" for case in clean):
        raise RuntimeError("source must contain only marked synthetic paired-clean previews")
    by_time: dict[float, list[dict]] = {}
    for case in clean:
        by_time.setdefault(float(case["t"]), []).append(case)

    requests = request_catalog()
    lookup = {request["request_id"]: request for request in requests}
    cases: list[dict] = []
    for t_index, (time_value, pool) in enumerate(sorted(by_time.items(), reverse=True)):
        cursor = 0
        for repeat in range(args.threshold_replicates):
            base = pool[cursor % len(pool)]; cursor += 1
            age_row = dict(base["record"])
            set_string(age_row, "education", "Doctorate")
            age_row["education.num"] = 16.0
            age_row["age"] = 24.0
            row_id = f"age_row_t{t_index}_r{repeat}"
            pair_id = f"age_swap_t{t_index}_r{repeat}"
            for request_id in ("degree_age_18", "degree_age_30"):
                add_case(cases, base, lookup[request_id], dict(age_row), row_id, pair_id, "same_row_threshold_swap")

            base = pool[cursor % len(pool)]; cursor += 1
            hours_row = dict(base["record"])
            set_string(hours_row, "workclass", "Private")
            hours_row["hours.per.week"] = 35.0
            row_id = f"hours_row_t{t_index}_r{repeat}"
            pair_id = f"hours_swap_t{t_index}_r{repeat}"
            for request_id in ("private_hours_25", "private_hours_45"):
                add_case(cases, base, lookup[request_id], dict(hours_row), row_id, pair_id, "same_row_threshold_swap")

        for repeat in range(args.control_replicates):
            spouse_request = lookup["spouse_marital_with_af_exception"]
            spouse_specs = [
                ("excepted", "Husband", "Married-AF-spouse"),
                ("satisfied", "Wife", "Married-civ-spouse"),
                ("violated", "Husband", "Never-married"),
                ("not_applicable", "Own-child", "Never-married"),
            ]
            for expected, relationship, marital in spouse_specs:
                base = pool[cursor % len(pool)]; cursor += 1
                row = dict(base["record"])
                set_string(row, "relationship", relationship)
                set_string(row, "marital.status", marital)
                add_case(
                    cases, base, spouse_request, row,
                    f"spouse_{expected}_t{t_index}_r{repeat}", None, f"explicit_{expected}",
                )
                if cases[-1]["gold_status"] != expected:
                    raise AssertionError((cases[-1], expected))

            missing_request = lookup["occupation_missing_with_never_worked_exception"]
            missing_specs = [
                ("excepted", "?", "Never-worked"),
                ("satisfied", "?", "?"),
                ("violated", "?", "Private"),
                ("not_applicable", "Craft-repair", "Private"),
            ]
            for expected, occupation, workclass in missing_specs:
                base = pool[cursor % len(pool)]; cursor += 1
                row = dict(base["record"])
                set_string(row, "occupation", occupation)
                set_string(row, "workclass", workclass)
                add_case(
                    cases, base, missing_request, row,
                    f"missing_{expected}_t{t_index}_r{repeat}", None, f"explicit_{expected}",
                )
                if cases[-1]["gold_status"] != expected:
                    raise AssertionError((cases[-1], expected))

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
        "status_counts": Counter(case["gold_status"] for case in cases),
        "swap_pairs": len({case["pair_id"] for case in cases if case["pair_id"]}),
        "source_policy": "Only paired-clean synthetic TabDiff previews; no real rows.",
        "leakage_policy": "Gold labels and oracle DSL stay local; API runners replace all IDs with opaque randomized IDs.",
    }
    # Counter is JSON-compatible as a dict subclass, but normalize for clarity.
    metadata["status_counts"] = dict(metadata["status_counts"])
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
