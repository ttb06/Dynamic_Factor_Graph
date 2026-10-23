#!/usr/bin/env python3
"""Aggregate matched full-sample component-planner experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--job-id",required=True)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    control=pd.read_csv(args.control/"method_metrics.csv")
    for method in ["unconditional","final_reward_calibrated","x0_reward_calibrated"]:
        row=control[control.method==method].iloc[0].to_dict()
        rows.append({"profile":f"edge_local_{method}",**row})
    for path in sorted(args.root.glob(f"component_planner_full_*_{args.job_id}_*")):
        metrics=pd.read_csv(path/"method_metrics.csv")
        metadata=json.loads((path/"run_metadata.json").read_text())
        profile=path.name.split(f"_{args.job_id}_")[0].replace("component_planner_full_","")
        for method in ["x0_reward_calibrated","x0_reward_calibrated_then_final_reward"]:
            match=metrics[metrics.method==method]
            if match.empty: continue
            row=match.iloc[0].to_dict()
            trace_path=path/f"trace_{method}.json"
            trace=json.loads(trace_path.read_text()) if trace_path.exists() else {}
            rows.append({"profile":f"component_{profile}_{'online' if method=='x0_reward_calibrated' else 'certified'}",
                         **row,"resolved_edge_events":trace.get("resolved_edge_events",0),
                         "changed_cells":trace.get("changed_cells",0),
                         "graph_weight":metadata["graph_weight"],
                         "guidance_scale":metadata["numeric_guidance_scale"],
                         "violation_weight":metadata["component_violation_weight"]})
    table=pd.DataFrame(rows)
    table.to_csv(args.output/"all_results.csv",index=False)
    columns=["profile","macro_violation","heldout_violation_error","calibration_violation_error",
             "actuate_heldout_violation_error","score_only_heldout_violation_error",
             "monitor_heldout_violation_error","numeric_ks","categorical_tv","action_events",
             "unique_rows_actuated","runtime_seconds"]
    lines=["# Connected-component graph planner", "",
           "Matched Shoppers checkpoint, seed, sample count, graph, and evaluation graph.", "",
           "| Profile | Macro | Heldout | Calibration | Actuate | Score-only | Monitor | KS | TV | Actions | Rows | Runtime |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in table[columns].itertuples(index=False):
        lines.append(f"| {row.profile} | {row.macro_violation:.5f} | {row.heldout_violation_error:.5f} | "
                     f"{row.calibration_violation_error:.5f} | {row.actuate_heldout_violation_error:.5f} | "
                     f"{row.score_only_heldout_violation_error:.5f} | {row.monitor_heldout_violation_error:.5f} | "
                     f"{row.numeric_ks:.5f} | {row.categorical_tv:.5f} | {int(row.action_events)} | "
                     f"{int(row.unique_rows_actuated)} | {row.runtime_seconds:.1f} |")
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
