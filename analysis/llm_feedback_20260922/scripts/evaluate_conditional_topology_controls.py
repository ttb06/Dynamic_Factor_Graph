#!/usr/bin/env python3
"""Paired audit for controlled row-conditional constraint experiments.

The generic graph metric only reports conditional violation.  This audit also
checks whether a sampler satisfies an implication by changing its antecedent,
and whether it edits target columns on rows that were outside the requested
scope in the paired unconditional sample.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DIR_RE = re.compile(
    r"conditional_topology_(?:locked_)?(?P<request>range|category|both)_"
    r"(?P<variant>dynamic|frozen|shuffled|global)_(?P<job>\d+)_(?P<task>\d+)"
)
METHOD_FILES = {
    "unconditional": "unconditional.csv",
    "online": "xstate_reward_calibrated.csv",
    "online+certifier": "xstate_reward_calibrated_then_final_reward.csv",
    "final-only": "final_reward_calibrated.csv",
}


def applicability(frame: pd.DataFrame) -> np.ndarray:
    return frame["MARRIAGE"].astype(str).to_numpy() == "1"


def requirement_ok(frame: pd.DataFrame, request: str) -> np.ndarray:
    ok = np.ones(len(frame), dtype=bool)
    if request in {"range", "both"}:
        ok &= pd.to_numeric(frame["AGE"], errors="coerce").to_numpy() >= 50.0
    if request in {"category", "both"}:
        ok &= frame["EDUCATION"].astype(str).to_numpy() == "3"
    return ok


def changed(a: pd.Series, b: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return ~np.isclose(
            pd.to_numeric(a, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(b, errors="coerce").to_numpy(dtype=float),
            atol=1e-8,
            equal_nan=True,
        )
    return a.astype(str).to_numpy() != b.astype(str).to_numpy()


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def audit_sample(base: pd.DataFrame, sample: pd.DataFrame, request: str) -> dict:
    if len(base) != len(sample):
        raise ValueError("Paired samples have different row counts")
    base_app = applicability(base)
    sample_app = applicability(sample)
    ok = requirement_ok(sample, request)
    violation = sample_app & ~ok

    target_columns = []
    if request in {"range", "both"}:
        target_columns.append("AGE")
    if request in {"category", "both"}:
        target_columns.append("EDUCATION")
    target_changed = np.zeros(len(base), dtype=bool)
    for column in target_columns:
        target_changed |= changed(base[column], sample[column])
    antecedent_changed = changed(base["MARRIAGE"], sample["MARRIAGE"])

    return {
        "rows": len(base),
        "applicable_support": int(sample_app.sum()),
        "applicable_rate": safe_mean(sample_app),
        "violation_overall": safe_mean(violation),
        "violation_given_applicable": safe_mean((~ok)[sample_app]),
        "antecedent_changed": safe_mean(antecedent_changed),
        "antecedent_evasion_given_base_applicable": safe_mean((~sample_app)[base_app]),
        "antecedent_creation_given_base_nonapplicable": safe_mean(sample_app[~base_app]),
        "target_changed_all": safe_mean(target_changed),
        "target_changed_given_base_nonapplicable": safe_mean(target_changed[~base_app]),
        "target_changed_given_base_applicable": safe_mean(target_changed[base_app]),
        "any_control_column_changed": safe_mean(target_changed | antecedent_changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    pattern = f"conditional_topology_*_{args.job_id}_*"
    for directory in sorted(args.root.glob(pattern)):
        match = DIR_RE.fullmatch(directory.name)
        if not match:
            continue
        request, variant = match.group("request"), match.group("variant")
        base = pd.read_csv(directory / "samples" / METHOD_FILES["unconditional"])
        generic = pd.read_csv(directory / "method_metrics.csv").set_index("method")
        trace = json.loads((directory / "trace_xstate_reward_calibrated.json").read_text())
        for method, filename in METHOD_FILES.items():
            sample = pd.read_csv(directory / "samples" / filename)
            row = {
                "request": request,
                "variant": variant,
                "method": method,
                "directory": str(directory),
                **audit_sample(base, sample, request),
            }
            generic_method = {
                "unconditional": "unconditional",
                "online": "xstate_reward_calibrated",
                "online+certifier": "xstate_reward_calibrated_then_final_reward",
                "final-only": "final_reward_calibrated",
            }[method]
            metrics = generic.loc[generic_method]
            for key in ["numeric_ks", "categorical_tv", "numeric_drift_sd",
                        "action_events", "unique_rows_actuated", "runtime_seconds"]:
                row[key] = float(metrics[key])
            if method == "online":
                choices = trace.get("action_choices", {})
                row["deactivation_actions"] = int(sum(
                    value for key, value in choices.items() if key.startswith("deactivate_")
                ))
                row["direct_requirement_actions"] = int(choices.get("direct_requirements", 0))
                row["topology_flips"] = int(trace.get("topology_flips", 0) or 0)
            else:
                row["deactivation_actions"] = 0
                row["direct_requirement_actions"] = 0
                row["topology_flips"] = 0
            rows.append(row)

    result = pd.DataFrame(rows).sort_values(["request", "method", "variant"])
    result.to_csv(args.output / "paired_audit.csv", index=False)

    online = result[result.method == "online"].copy()
    online.to_csv(args.output / "online_comparison.csv", index=False)
    columns = [
        "request", "variant", "violation_given_applicable",
        "antecedent_evasion_given_base_applicable",
        "target_changed_given_base_nonapplicable", "applicable_rate",
        "numeric_ks", "categorical_tv", "action_events", "topology_flips",
    ]
    lines = [
        "# Conditional-topology paired audit", "",
        "All rates use paired samples with identical seed/noise. `Violation` is",
        "computed only among rows to which the condition applies in the final sample.",
        "`Scope loss` measures rows originally in scope that leave scope by",
        "changing `MARRIAGE` (a paired diagnostic, not automatically intentional evasion);",
        "`outside-target edits` measures target edits on rows",
        "that were outside scope in the unconditional counterpart.", "",
        "| Request | Variant | Violation in scope | Scope loss | Outside-target edits | Final scope | Numeric KS | Categorical TV | Actions | Topology flips |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in online[columns].itertuples(index=False):
        lines.append(
            f"| {item.request} | {item.variant} | {item.violation_given_applicable:.2%} | "
            f"{item.antecedent_evasion_given_base_applicable:.2%} | "
            f"{item.target_changed_given_base_nonapplicable:.2%} | "
            f"{item.applicable_rate:.2%} | {item.numeric_ks:.4f} | "
            f"{item.categorical_tv:.4f} | {int(item.action_events)} | {int(item.topology_flips)} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
