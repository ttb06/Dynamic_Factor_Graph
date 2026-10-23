#!/usr/bin/env python3
"""Paired evaluator for a graph whose second edge emerges after first-edge action."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def masks(frame: pd.DataFrame):
    scope = frame["MARRIAGE"].astype(str).eq("1").to_numpy()
    age_ok = pd.to_numeric(frame["AGE"], errors="coerce").ge(50).to_numpy()
    education_ok = frame["EDUCATION"].astype(str).eq("3").to_numpy()
    return scope, age_ok, education_ok


def rate(values):
    return float(np.mean(values)) if len(values) else float("nan")


def evaluate(base, sample):
    base_scope, _, _ = masks(base)
    scope, age_ok, education_ok = masks(sample)
    changed_age = ~np.isclose(pd.to_numeric(base.AGE), pd.to_numeric(sample.AGE))
    changed_education = base.EDUCATION.astype(str).to_numpy() != sample.EDUCATION.astype(str).to_numpy()
    return {
        "chain_violation_given_scope": rate((~(age_ok & education_ok))[scope]),
        "age_edge_violation_given_scope": rate((~age_ok)[scope]),
        "education_edge_violation_given_active": rate((~education_ok)[scope & age_ok]),
        "scope_loss": rate((~scope)[base_scope]),
        "target_edit_outside_base_scope": rate((changed_age | changed_education)[~base_scope]),
        "age_edit_outside_base_scope": rate(changed_age[~base_scope]),
        "education_edit_outside_base_scope": rate(changed_education[~base_scope]),
        "final_scope": rate(scope),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for variant in ["dynamic", "frozen", "late_only"]:
        directory = next(args.root.glob(f"controlled_chain_{variant}_{args.job_id}_*"))
        base = pd.read_csv(directory / "samples/unconditional.csv")
        metrics = pd.read_csv(directory / "method_metrics.csv").set_index("method")
        trace = json.loads((directory / "trace_xstate_reward_calibrated.json").read_text())
        for method, filename in [
            ("online", "xstate_reward_calibrated.csv"),
            ("online+certifier", "xstate_reward_calibrated_then_final_reward.csv"),
            ("final-only", "final_reward_calibrated.csv"),
        ]:
            sample = pd.read_csv(directory / "samples" / filename)
            generic_name = {
                "online": "xstate_reward_calibrated",
                "online+certifier": "xstate_reward_calibrated_then_final_reward",
                "final-only": "final_reward_calibrated",
            }[method]
            generic = metrics.loc[generic_name]
            row = {"variant": variant, "method": method, **evaluate(base, sample),
                   "numeric_ks": float(generic.numeric_ks),
                   "categorical_tv": float(generic.categorical_tv),
                   "action_events": int(generic.action_events),
                   "directory": str(directory)}
            if method == "online":
                row["age_actions"] = int(trace.get("actions_by_edge", {}).get("chain_age", 0))
                row["education_actions"] = int(trace.get("actions_by_edge", {}).get("chain_education", 0))
                row["age_topology_changes"] = int(trace.get("topology_changes_by_edge", {}).get("chain_age", 0))
                row["education_topology_changes"] = int(trace.get("topology_changes_by_edge", {}).get("chain_education", 0))
            else:
                row.update(age_actions=0, education_actions=0,
                           age_topology_changes=0, education_topology_changes=0)
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "controlled_chain_metrics.csv", index=False)
    online = result[result.method == "online"]
    lines = [
        "# Controlled dynamic-chain result", "",
        "| Variant | Chain violation | Age-edge violation | Education-edge violation | Scope loss | Outside edit | KS | TV | Age actions | Education actions | Education topology changes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in online.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.chain_violation_given_scope:.2%} | "
            f"{row.age_edge_violation_given_scope:.2%} | {row.education_edge_violation_given_active:.2%} | "
            f"{row.scope_loss:.2%} | {row.target_edit_outside_base_scope:.2%} | "
            f"{row.numeric_ks:.4f} | {row.categorical_tv:.4f} | {row.age_actions} | "
            f"{row.education_actions} | {row.education_topology_changes} |"
        )
    lines += ["", "The graph is a controlled mechanism probe, not a discovered semantic invariant."]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
