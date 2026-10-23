#!/usr/bin/env python3
"""Aggregate pretrained, continued-training, and graph-finetuned TabDiff pilots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ["pretrained", "continue", "graph1_mid", "graph5_mid", "graph20_mid", "graph5_uniform"]
METHODS = ["unconditional", "x0_reward_calibrated", "x0_reward_calibrated_then_final_reward", "final_reward_calibrated"]


def gain(reference: float, value: float) -> float:
    return 100*(reference-value)/reference if reference else np.nan


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--train-job",required=True); parser.add_argument("--trajectory-job",required=True)
    parser.add_argument("--evaluation-job",required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    outputs=args.analysis/"outputs"; rows=[]
    for index,variant in enumerate(VARIANTS):
        metrics_path=outputs/f"e2e_graph_eval_{variant}_{args.evaluation_job}_{index}"/"method_metrics.csv"
        trajectory_path=outputs/f"e2e_graph_trajectory_{variant}_{args.trajectory_job}_{index}"/"typed_readability_schedule.csv"
        metrics=pd.read_csv(metrics_path); schedule=pd.read_csv(trajectory_path)
        training={"graph_weight":0.0,"relative_parameter_displacement":0.0,"training_runtime_seconds":0.0}
        if index>0:
            train_index=index-1
            metadata_path=outputs/f"e2e_graph_finetune_{variant}_{args.train_job}_{train_index}"/"training_metadata.json"
            training=json.loads(metadata_path.read_text())
        for method in METHODS:
            selected=metrics[metrics.method.eq(method)]
            if selected.empty: continue
            row=selected.iloc[0]
            rows.append({
                "variant":variant,"method":method,
                "graph_weight":training.get("graph_weight",0.0),
                "parameter_displacement":training.get("relative_parameter_displacement",0.0),
                "training_runtime_seconds":training.get("runtime_seconds",0.0),
                "readable_edges":int(schedule.readable.sum()),
                "mean_readability_cutoff":float(schedule.loc[schedule.readable,"cutoff"].mean()),
                "median_readability_cutoff":float(schedule.loc[schedule.readable,"cutoff"].median()),
                **{column:row[column] for column in [
                    "macro_violation","heldout_violation_error","calibration_violation_error",
                    "actuate_macro_violation","actuate_heldout_violation_error","numeric_ks","categorical_tv",
                    "numeric_drift_sd","action_events","unique_rows_actuated","runtime_seconds",
                ]},
            })
    result=pd.DataFrame(rows)
    for metric in ("macro_violation","heldout_violation_error","calibration_violation_error"):
        result[f"{metric}_gain_vs_pretrained_pct"]=np.nan
        for method,block in result.groupby("method"):
            base=block[block.variant.eq("pretrained")]
            if base.empty: continue
            ref=float(base.iloc[0][metric]); result.loc[block.index,f"{metric}_gain_vs_pretrained_pct"]=[gain(ref,float(v)) for v in block[metric]]
    result.to_csv(args.output/"e2e_graph_results.csv",index=False)

    primary=result[result.method.isin(["unconditional","x0_reward_calibrated"])].copy()
    lines=["# End-to-end graph-consistent TabDiff pilot","",
           "All variants use the same pretrained initialization, train split, evaluator and paired evaluation seed. `continue` is the extra-training control.","",
           "| Variant | Method | Graph λ | Macro | Heldout error | Calibration error | Gain macro vs pretrained | Num. KS | Cat. TV | Mean readable t | Param. displacement | Actions |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in primary.itertuples(index=False):
        lines.append(f"| {row.variant} | {row.method} | {row.graph_weight:g} | {row.macro_violation:.5f} | "
                     f"{row.heldout_violation_error:.5f} | {row.calibration_violation_error:.5f} | "
                     f"{row.macro_violation_gain_vs_pretrained_pct:+.1f}% | {row.numeric_ks:.5f} | {row.categorical_tv:.5f} | "
                     f"{row.mean_readability_cutoff:.3f} | {row.parameter_displacement:.5f} | {int(row.action_events)} |")
    lines += ["", "Decision rule: graph fine-tuning must beat both pretrained and continued-training controls, preserve KS/TV, and either improve unconditional validity or make graph states readable earlier. If only guided output improves, training and inference guidance are complementary; if unconditional improves but guidance adds nothing, the training objective may replace the sampler intervention."]
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__=="__main__": main()
