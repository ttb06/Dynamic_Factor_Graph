#!/usr/bin/env python3
"""Aggregate paired score-only graph guardrail experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "actuate_heldout_violation_error", "score_only_heldout_violation_error",
    "heldout_violation_error", "numeric_ks", "categorical_tv", "numeric_drift_sd",
    "action_events", "runtime_seconds",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown(frame: pd.DataFrame) -> str:
    values=frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column]=values[column].map(lambda value:"" if pd.isna(value) else f"{value:.6f}")
    rows=["| "+" | ".join(map(str,row))+" |" for row in values.itertuples(index=False,name=None)]
    return "\n".join(["| "+" | ".join(values.columns)+" |","| "+" | ".join(["---"]*len(values.columns))+" |",*rows])


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--job",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    root=args.analysis/"outputs"; rows=[]; parity=[]; choices=[]
    for seed in ["20261061","20261062","20261063"]:
        directories={weight:next(root.glob(f"diabetes_graph_guard_w{weight}_s{seed}_{args.job}_*")) for weight in [0,2]}
        hashes={weight:digest(directory/"samples"/"unconditional.csv") for weight,directory in directories.items()}
        parity.append({"seed":seed,"hash_w0":hashes[0],"hash_w2":hashes[2],"equal":hashes[0]==hashes[2]})
        for weight,directory in directories.items():
            frame=pd.read_csv(directory/"method_metrics.csv")
            record=frame.loc[frame.method.eq("x0_reward_calibrated")].iloc[0].to_dict()
            trace=json.loads((directory/"trace_x0_reward_calibrated.json").read_text())
            rows.append({**record,"seed":seed,"graph_weight":weight})
            choices.append({"seed":seed,"graph_weight":weight,**trace.get("action_choices",{})})
    per_seed=pd.DataFrame(rows); parity=pd.DataFrame(parity); choice_frame=pd.DataFrame(choices).fillna(0)
    per_seed.to_csv(args.output/"guardrail_per_seed.csv",index=False)
    parity.to_csv(args.output/"base_parity.csv",index=False)
    choice_frame.to_csv(args.output/"action_choices.csv",index=False)
    summary=per_seed.groupby("graph_weight")[METRICS].agg(["mean","std"]).reset_index()
    summary.columns=["_".join(value).rstrip("_") if isinstance(value,tuple) else value for value in summary.columns]
    summary.to_csv(args.output/"guardrail_summary.csv",index=False)
    view=per_seed.pivot(index="seed",columns="graph_weight")
    lines=["# Score-only graph guardrail", "",
           f"Unconditional parity passed: {int(parity.equal.sum())}/{len(parity)} seeds.", "",
           markdown(summary), "",
           f"Graph guardrail lowers score-only heldout error in {int((view['score_only_heldout_violation_error'][2] < view['score_only_heldout_violation_error'][0]).sum())}/3 seeds.",
           f"Graph guardrail lowers overall heldout error in {int((view['heldout_violation_error'][2] < view['heldout_violation_error'][0]).sum())}/3 seeds.", ""]
    (args.output/"RESULTS.md").write_text("\n".join(lines))


if __name__=="__main__": main()
