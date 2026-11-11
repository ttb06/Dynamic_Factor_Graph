"""Runtime helpers extracted from the experiment implementation."""
from __future__ import annotations

import math


import numpy as np


import pandas as pd


from dygraft.predicates import predicate_mask


def edge_masks(frame: pd.DataFrame, edge: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.ones(len(frame), dtype=bool)
    for atom in edge.get("applicability", []): active &= predicate_mask(frame, atom)
    exception = np.zeros(len(frame), dtype=bool)
    for group in edge.get("exceptions", []):
        group_mask = np.ones(len(frame), dtype=bool)
        for atom in group.get("all_of", []): group_mask &= predicate_mask(frame, atom)
        exception |= group_mask
    active &= ~exception
    satisfied = np.ones(len(frame), dtype=bool)
    for atom in edge.get("requirements", []): satisfied &= predicate_mask(frame, atom)
    return active, satisfied, active & ~satisfied


def graph_metrics(frame: pd.DataFrame, reference: pd.DataFrame, edges: list[dict]) -> dict:
    rates=[]; heldout_errors=[]; calibration_errors=[]; total_support=total_bad=0
    for edge in edges:
        active,_,bad=edge_masks(frame,edge); ractive,_,rbad=edge_masks(reference,edge)
        if active.sum():
            rate=float(bad.sum()/active.sum()); rates.append(rate); total_support+=int(active.sum()); total_bad+=int(bad.sum())
            if ractive.sum(): heldout_errors.append(abs(rate-float(rbad.sum()/ractive.sum())))
            calibration_errors.append(abs(rate-float(edge["calibration"]["violation_rate"])))
    return {"macro_violation":float(np.mean(rates)) if rates else math.nan,
        "micro_violation":total_bad/total_support if total_support else math.nan,
        "heldout_violation_error":float(np.mean(heldout_errors)) if heldout_errors else math.nan,
        "calibration_violation_error":float(np.mean(calibration_errors)) if calibration_errors else math.nan}
