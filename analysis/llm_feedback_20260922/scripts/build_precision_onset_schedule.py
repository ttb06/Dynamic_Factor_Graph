#!/usr/bin/env python3
"""Build a precision-first edge onset schedule from a calibration trajectory.

The original readability schedule requires both precision and recall.  For an
online selective action, low recall is not itself unsafe: rows not yet visible
can be reconsidered later.  This schedule therefore opens an edge when the
currently visible violations have enough support and a conservative Wilson
precision lower bound, while retaining the reference schedule for edges whose
final violation state was not evaluable.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def wilson_lower(successes: float, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = min(1.0, max(0.0, float(successes) / total))
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (centre - radius) / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--reference-schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-current-violations", type=int, default=12)
    parser.add_argument("--min-active-precision", type=float, default=.90)
    parser.add_argument("--min-precision-lower", type=float, default=.65)
    parser.add_argument("--roles", nargs="+", default=["actuate"],
                        help="Only change onset for these graph roles; retain all other reference cutoffs.")
    args = parser.parse_args()

    trajectory = pd.read_csv(args.trajectory)
    schedule = pd.read_csv(args.reference_schedule)
    new_cutoffs: dict[str, float] = {}
    audit_rows = []
    for row in schedule.itertuples(index=False):
        edge_id = str(row.candidate_id)
        original = float(row.cutoff) if pd.notna(row.cutoff) else math.nan
        block = trajectory[trajectory.candidate_id.astype(str).eq(edge_id)].copy()
        block["precision_successes"] = block.violation_precision.fillna(0) * block.violation.fillna(0)
        block["precision_lower"] = [
            wilson_lower(successes, int(total))
            for successes, total in zip(block.precision_successes, block.violation.fillna(0))
        ]
        if str(row.usage_role) not in set(args.roles):
            cutoff = original
            source = "reference_unchanged_role"
        elif bool(row.violation_evaluable):
            eligible = block[
                block.active_precision.fillna(0).ge(args.min_active_precision)
                & block.violation.fillna(0).ge(args.min_current_violations)
                & block.precision_lower.ge(args.min_precision_lower)
            ]
            cutoff = float(eligible.t.max()) if not eligible.empty else original
            source = "precision_onset" if not eligible.empty else "reference_fallback"
        else:
            cutoff = original
            source = "reference_not_evaluable"
        new_cutoffs[edge_id] = cutoff
        audit_rows.append({
            "candidate_id": edge_id,
            "reference_cutoff": original,
            "precision_onset_cutoff": cutoff,
            "delta": cutoff - original if math.isfinite(cutoff) and math.isfinite(original) else math.nan,
            "source": source,
        })

    schedule["cutoff"] = schedule.candidate_id.astype(str).map(new_cutoffs)
    schedule["readable"] = schedule.cutoff.notna()
    args.output.mkdir(parents=True, exist_ok=False)
    schedule.to_csv(args.output / "typed_readability_schedule.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output / "onset_audit.csv", index=False)
    (args.output / "RESULTS.md").write_text(
        "# Precision-onset trajectory schedule\n\n"
        f"Minimum current violations: {args.min_current_violations}; "
        f"minimum active precision: {args.min_active_precision:.2f}; "
        f"minimum Wilson precision lower bound: {args.min_precision_lower:.2f}; "
        f"changed roles: {', '.join(args.roles)}.\n\n"
        "The machine-readable audit is in `onset_audit.csv`.\n"
    )


if __name__ == "__main__":
    main()
