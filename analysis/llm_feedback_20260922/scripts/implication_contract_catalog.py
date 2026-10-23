#!/usr/bin/env python3
"""Conditional-implication contracts for the applicability/evasion pilot."""

from __future__ import annotations

import pandas as pd


ADVANCED = {"Masters", "Prof-school", "Doctorate"}


def make(
    contract_id: str, split: str, antecedents: tuple[str, ...], target: str,
    threshold: float, *, exception_exec: bool = False,
) -> dict:
    antecedent_text = " and ".join(
        {"advanced": "education is Masters, Prof-school, or Doctorate", "private": "workclass is Private"}[item]
        for item in antecedents
    )
    exception = " Exception: occupation Exec-managerial is allowed unchanged." if exception_exec else ""
    return {
        "contract_id": contract_id,
        "split": split,
        "antecedents": list(antecedents),
        "target": target,
        "threshold": float(threshold),
        "exception_exec": exception_exec,
        "text": f"If {antecedent_text}, {target} must be at least {threshold:g}.{exception}",
    }


def catalog() -> list[dict]:
    train: list[dict] = []
    for antecedent in ("advanced", "private"):
        for target in ("age", "hours.per.week"):
            for threshold in (35, 50):
                train.append(make(f"{antecedent}_{target}_ge_{threshold}", "train", (antecedent,), target, threshold))
    train.extend(
        [
            make("private_hours_ge_35_unless_exec", "train", ("private",), "hours.per.week", 35, exception_exec=True),
            make("private_hours_ge_50_unless_exec", "train", ("private",), "hours.per.week", 50, exception_exec=True),
        ]
    )
    heldout = [
        make("advanced_age_ge_45", "heldout_threshold", ("advanced",), "age", 45),
        make("advanced_hours_ge_45", "heldout_threshold", ("advanced",), "hours.per.week", 45),
        make("private_age_ge_45", "heldout_threshold", ("private",), "age", 45),
        make("private_hours_ge_45", "heldout_threshold", ("private",), "hours.per.week", 45),
        make("private_hours_ge_45_unless_exec", "heldout_exception", ("private",), "hours.per.week", 45, exception_exec=True),
        make("advanced_private_age_ge_45", "heldout_composition", ("advanced", "private"), "age", 45),
        make("advanced_private_hours_ge_45", "heldout_composition", ("advanced", "private"), "hours.per.week", 45),
        make("advanced_age_ge_55", "heldout_extrapolation", ("advanced",), "age", 55),
        make("private_hours_ge_55", "heldout_extrapolation", ("private",), "hours.per.week", 55),
    ]
    return train + heldout


def vector(contract: dict) -> list[float]:
    return [
        float("advanced" in contract["antecedents"]),
        float("private" in contract["antecedents"]),
        float(contract["target"] == "age"),
        float(contract["target"] == "hours.per.week"),
        float(contract["threshold"]) / 100.0,
        float(contract["exception_exec"]),
    ]


def antecedent_mask(frame: pd.DataFrame, contract: dict) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if "advanced" in contract["antecedents"]:
        mask &= frame["education"].astype(str).str.strip().isin(ADVANCED)
    if "private" in contract["antecedents"]:
        mask &= frame["workclass"].astype(str).str.strip().eq("Private")
    return mask


def exception_mask(frame: pd.DataFrame, contract: dict) -> pd.Series:
    if not contract["exception_exec"]:
        return pd.Series(False, index=frame.index)
    return antecedent_mask(frame, contract) & frame["occupation"].astype(str).str.strip().eq("Exec-managerial")


def violation_mask(frame: pd.DataFrame, contract: dict) -> pd.Series:
    active = antecedent_mask(frame, contract) & ~exception_mask(frame, contract)
    return active & pd.to_numeric(frame[contract["target"]], errors="coerce").lt(contract["threshold"])

