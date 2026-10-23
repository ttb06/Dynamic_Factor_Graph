#!/usr/bin/env python3
"""Aggregate temporal-hysteresis topology runs against dynamic references."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

METRICS=["actuate_heldout_violation_error","heldout_violation_error","numeric_ks","categorical_tv","numeric_drift_sd","action_events","runtime_seconds"]
SEEDS=["20261031","20261032","20261033"]
TASKS={"shoppers":[0,1,2],"default":[6,7,8]}

def select(path):
    frame=pd.read_csv(path)
    return frame.loc[frame.method.eq("x0_reward_calibrated")].iloc[0].to_dict()

def markdown(frame):
    values=frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]): values[column]=values[column].map(lambda x:f"{x:.6f}")
    rows=[[str(x) for x in row] for row in values.itertuples(index=False,name=None)]
    return "\n".join(["| "+" | ".join(map(str,values.columns))+" |","| "+" | ".join(["---"]*len(values.columns))+" |",*("| "+" | ".join(row)+" |" for row in rows)])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--analysis",type=Path,required=True); p.add_argument("--job-id",required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); root=a.analysis/"outputs"; rows=[]
    for dataset in ["shoppers","default"]:
        for ordinal,seed in enumerate(SEEDS):
            task=TASKS[dataset][ordinal]
            for variant in ["dynamic","frozen_state"]:
                path=root/f"time_seed_{dataset}_middle_s{seed}_{variant}_1060317_{task}"/"method_metrics.csv"
                rows.append({**select(path),"dataset":dataset,"seed":seed,"method":variant})
            for persistence in [2,3]:
                path=next(root.glob(f"hysteresis_p{persistence}_{dataset}_s{seed}_{a.job_id}_*/method_metrics.csv"))
                rows.append({**select(path),"dataset":dataset,"seed":seed,"method":f"hysteresis_p{persistence}"})
    per_seed=pd.DataFrame(rows); per_seed.to_csv(a.output/"per_seed.csv",index=False)
    summary=per_seed.groupby(["dataset","method"])[METRICS].agg(["mean","std"]).reset_index()
    summary.columns=["_".join(x).rstrip("_") if isinstance(x,tuple) else x for x in summary.columns]
    summary.to_csv(a.output/"summary.csv",index=False)
    columns=["dataset","method","actuate_heldout_violation_error_mean","actuate_heldout_violation_error_std","heldout_violation_error_mean","numeric_ks_mean","categorical_tv_mean","action_events_mean","runtime_seconds_mean"]
    (a.output/"RESULTS.md").write_text("# Temporal-hysteresis ablation\n\n"+markdown(summary[columns])+"\n")

if __name__=="__main__": main()
