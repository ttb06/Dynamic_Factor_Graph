#!/usr/bin/env python3
"""Select per-row among paired middle-guidance trajectories using a train-only graph.

All candidate trajectories must share the exact unconditional sample.  The
guidance graph ranks candidates; a separate frozen evaluation graph is used
only after selection.  This is an intentionally compute-heavy diagnostic for
whether row-specific guidance strength is worth an online controller.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conditioned_tabdiff_adapter import decode
from rowwise_lookahead_branch import selection_scores, select_rows
from rowwise_reverse_action_guidance import RowActionLayer, edge_metric_rows, evaluate, final_action
from tabdiff_probe import preprocess_dataset


def parse_weights(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--graph",type=Path,required=True)
    parser.add_argument("--evaluation-graph",type=Path,required=True)
    parser.add_argument("--input-glob",action="append",required=True)
    parser.add_argument("--edit-weights",default="0.1,0.5,1.0,2.0")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)

    sys.path.insert(0,str(args.repo))
    info=json.loads((args.repo/"data"/args.dataset/"info.json").read_text())
    train_data,test_data=preprocess_dataset(args.repo,args.dataset)
    train=decode(train_data,train_data.X,info); heldout=decode(train_data,test_data.X,info)
    graph=json.loads(args.graph.read_text()); evaluation=json.loads(args.evaluation_graph.read_text())
    layer=RowActionLayer(graph,train_data,train,info,1.0,.5,1.0)

    directories=[]
    for pattern in args.input_glob:
        directories.extend(Path(item) for item in glob.glob(pattern))
    directories=sorted(set(directories))
    directories=[item for item in directories if (item/"samples/unconditional.csv").exists()
                 and (item/"samples/x0_reward_calibrated.csv").exists()]
    if len(directories)<2:
        raise RuntimeError(f"need at least two complete paired trajectories, found {len(directories)}")

    base=pd.read_csv(directories[0]/"samples/unconditional.csv")
    reference_bytes=(directories[0]/"samples/unconditional.csv").read_bytes()
    frames=[base]; names=["unconditional"]
    for directory in directories:
        if (directory/"samples/unconditional.csv").read_bytes()!=reference_bytes:
            raise RuntimeError(f"unconditional sample mismatch: {directory}")
        frames.append(pd.read_csv(directory/"samples/x0_reward_calibrated.csv"))
        names.append(directory.name)

    rows=[evaluate(base,base,heldout,evaluation,info,"unconditional",{},layer)]
    edge_rows=edge_metric_rows(base,heldout,evaluation,"unconditional")
    for name,frame in zip(names[1:],frames[1:]):
        changed=np.flatnonzero((frame.astype(str)!=base.astype(str)).any(axis=1)).astype(int).tolist()
        stats={"action_events":len(changed),"unique_rows_actuated":len(changed),"acted_rows":changed,
               "runtime_seconds":0}
        rows.append(evaluate(frame,base,heldout,evaluation,info,name,stats,layer))
        edge_rows+=edge_metric_rows(frame,heldout,evaluation,name)

    sample_dir=args.output/"samples"; sample_dir.mkdir()
    base.to_csv(sample_dir/"unconditional.csv",index=False)
    selections={}
    for edit_weight in parse_weights(args.edit_weights):
        scores=selection_scores(frames,base,graph,layer,edit_weight=edit_weight)
        selected,chosen=select_rows(frames,scores)
        method=f"paired_portfolio_edit_{edit_weight:g}".replace(".","p")
        changed=np.flatnonzero(chosen!=0).astype(int).tolist()
        stats={"action_events":len(changed),"unique_rows_actuated":len(changed),"acted_rows":changed,
               "runtime_seconds":0}
        rows.append(evaluate(selected,base,heldout,evaluation,info,method,stats,layer))
        edge_rows+=edge_metric_rows(selected,heldout,evaluation,method)
        selected.to_csv(sample_dir/f"{method}.csv",index=False)
        selections[method]={names[index]:int((chosen==index).sum()) for index in range(len(names))}

    repaired,repair_stats=final_action(base,graph,layer,"reward")
    rows.append(evaluate(repaired,base,heldout,evaluation,info,"final_reward_calibrated",repair_stats,layer))
    edge_rows+=edge_metric_rows(repaired,heldout,evaluation,"final_reward_calibrated")
    repaired.to_csv(sample_dir/"final_reward_calibrated.csv",index=False)

    metrics=pd.DataFrame(rows); metrics.to_csv(args.output/"method_metrics.csv",index=False)
    pd.DataFrame(edge_rows).to_csv(args.output/"edge_metrics.csv",index=False)
    diagnostics={"dataset":args.dataset,"candidate_count":len(frames)-1,
                 "candidate_directories":[str(item) for item in directories],
                 "selection_graph":str(args.graph),"evaluation_graph":str(args.evaluation_graph),
                 "selection_counts":selections,"evaluation_used_for_selection":False}
    (args.output/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    lines=["# Paired trajectory portfolio","",
           f"Candidates: `{len(frames)-1}`. Evaluation graph used for selection: `False`.","",
           "| Method | Violation | Heldout error | Calibration error | KS | TV | Drift | Rows selected |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.method} | {row.macro_violation:.5f} | {row.heldout_violation_error:.5f} | {row.calibration_violation_error:.5f} | {row.numeric_ks:.5f} | {row.categorical_tv:.5f} | {row.numeric_drift_sd:.5f} | {int(row.unique_rows_actuated)} |")
    lines.extend(["",f"Selection counts: `{json.dumps(selections,sort_keys=True)}`"])
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
