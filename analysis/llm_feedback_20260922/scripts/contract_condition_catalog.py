#!/usr/bin/env python3
"""Contract catalog for the conditional TabDiff adapter pilot.

The conditions are cohort-style semantic contracts over Adult numerical
features.  Train contracts contain only atomic predicates.  Evaluation adds
held-out thresholds and unseen conjunctions to test compositional transfer.
"""

from __future__ import annotations


SCALES = {"age": 100.0, "education.num": 16.0, "hours.per.week": 100.0}
COLUMNS = list(SCALES)


def item(contract_id: str, split: str, predicates: list[tuple[str, float]], text: str, family: str) -> dict:
    return {
        "contract_id": contract_id,
        "split": split,
        "family": family,
        "predicates": [{"column": column, "operator": "gte", "threshold": threshold} for column, threshold in predicates],
        "text": text,
    }


def catalog() -> list[dict]:
    train = [
        item("age_ge_25", "train", [("age", 25)], "Generate adults whose age is at least 25 years.", "age_floor"),
        item("age_ge_35", "train", [("age", 35)], "The generated record must have age no lower than 35.", "age_floor"),
        item("age_ge_50", "train", [("age", 50)], "Require a minimum age of fifty years.", "age_floor"),
        item("hours_ge_20", "train", [("hours.per.week", 20)], "Weekly working time must reach at least 20 hours.", "hours_floor"),
        item("hours_ge_35", "train", [("hours.per.week", 35)], "Generate a row with hours.per.week greater than or equal to 35.", "hours_floor"),
        item("hours_ge_50", "train", [("hours.per.week", 50)], "Require no fewer than fifty work hours per week.", "hours_floor"),
        item("edu_ge_9", "train", [("education.num", 9)], "Education level number must be at least 9.", "education_floor"),
        item("edu_ge_12", "train", [("education.num", 12)], "Generate education.num at or above twelve.", "education_floor"),
        item("edu_ge_14", "train", [("education.num", 14)], "Require an education level of 14 or higher.", "education_floor"),
    ]
    heldout = [
        item("age_ge_30", "heldout_threshold", [("age", 30)], "Only generate people aged thirty or older.", "age_floor"),
        item("age_ge_45", "heldout_threshold", [("age", 45)], "Age must have reached at least 45 years.", "age_floor"),
        item("hours_ge_30", "heldout_threshold", [("hours.per.week", 30)], "Set a floor of thirty on weekly work hours.", "hours_floor"),
        item("hours_ge_45", "heldout_threshold", [("hours.per.week", 45)], "The weekly-hours field must be 45 or above.", "hours_floor"),
        item("edu_ge_10", "heldout_threshold", [("education.num", 10)], "Education numeric level cannot be below ten.", "education_floor"),
        item("edu_ge_13", "heldout_threshold", [("education.num", 13)], "Generate education.num of at least 13.", "education_floor"),
    ]
    composition = [
        item(
            "age35_hours35", "heldout_composition", [("age", 35), ("hours.per.week", 35)],
            "Both conditions are mandatory: age is at least 35 and hours.per.week is at least 35.", "age_hours",
        ),
        item(
            "age30_hours45", "heldout_composition", [("age", 30), ("hours.per.week", 45)],
            "Generate a record jointly satisfying age >= 30 and weekly work hours >= 45.", "age_hours",
        ),
        item(
            "edu13_age30", "heldout_composition", [("education.num", 13), ("age", 30)],
            "The row needs education.num of 13 or more together with age of 30 or more.", "education_age",
        ),
    ]
    return train + heldout + composition


def structured_vector(contract: dict) -> list[float]:
    by_column = {predicate["column"]: float(predicate["threshold"]) for predicate in contract["predicates"]}
    vector: list[float] = []
    for column in COLUMNS:
        active = column in by_column
        vector.extend([1.0 if active else 0.0, by_column.get(column, 0.0) / SCALES[column]])
    return vector


def satisfies(record: dict, contract: dict) -> bool:
    return all(float(record[predicate["column"]]) >= float(predicate["threshold"]) for predicate in contract["predicates"])

