#!/usr/bin/env python3
"""Aggregate pre-registered scale-1.5 component-planner confirmation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def select(path: Path, method: str) -> dict:
    frame=pd.read_csv(path/"method_metrics.csv")
    return frame[frame.method==method].iloc[0].to_dict()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True)
    p.add_argument("--job-id",required=True); p.add_argument("--original-component",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    specs=[
        (20261301,a.root/f"selected_s15_edge_s20261301_{a.job_id}_0",a.original_component),
        (20261331,a.root/f"selected_s15_edge_s20261331_{a.job_id}_1",a.root/f"selected_s15_component_s20261331_{a.job_id}_2"),
        (20261332,a.root/f"selected_s15_edge_s20261332_{a.job_id}_3",a.root/f"selected_s15_component_s20261332_{a.job_id}_4"),
    ]
    rows=[]
    methods={"unconditional":"unconditional","final":"final_reward_calibrated",
             "edge":"x0_reward_calibrated","edge_certified":"x0_reward_calibrated_then_final_reward",
             "component":"x0_reward_calibrated","component_certified":"x0_reward_calibrated_then_final_reward"}
    for seed,edge_path,component_path in specs:
        for label in ["unconditional","final","edge","edge_certified"]:
            rows.append({"seed":seed,"profile":label,**select(edge_path,methods[label])})
        for label in ["component","component_certified"]:
            rows.append({"seed":seed,"profile":label,**select(component_path,methods[label])})
    data=pd.DataFrame(rows); data.to_csv(a.output/"per_seed.csv",index=False)
    metrics=["macro_violation","heldout_violation_error","calibration_violation_error","numeric_ks","categorical_tv","action_events"]
    summary=data.groupby("profile")[metrics].agg(["mean","std"]); summary.to_csv(a.output/"summary.csv")
    wide=data.pivot(index="seed",columns="profile",values="heldout_violation_error")
    gains=pd.DataFrame({
        "component_vs_edge_pct":100*(wide.edge-wide.component)/wide.edge,
        "component_certified_vs_final_pct":100*(wide.final-wide.component_certified)/wide.final,
    }); gains.to_csv(a.output/"paired_gains.csv")
    lines=["# Scale-1.5 component planner: three-seed confirmation","",
           "| Profile | Macro | Heldout | Calibration | KS | TV | Actions |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for profile in ["unconditional","final","edge","component","edge_certified","component_certified"]:
        block=data[data.profile==profile]
        vals=[]
        for metric in metrics:
            vals.append(f"{block[metric].mean():.5f} +/- {block[metric].std(ddof=1):.5f}")
        lines.append(f"| {profile} | "+" | ".join(vals)+" |")
    lines += ["",f"Component vs edge-local heldout gain: {gains.component_vs_edge_pct.mean():.2f}% +/- {gains.component_vs_edge_pct.std(ddof=1):.2f}%.",
              f"Component+certification vs final-only heldout gain: {gains.component_certified_vs_final_pct.mean():.2f}% +/- {gains.component_certified_vs_final_pct.std(ddof=1):.2f}%."]
    (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print("\n".join(lines))


if __name__=="__main__": main()
