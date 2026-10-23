#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--input-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--glob",default="mixed_condition_*_seed*/contract_metrics.csv")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    paths=sorted(args.input_root.glob(args.glob))
    if not paths: raise FileNotFoundError("no mixed-condition metrics")
    data=pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in paths],ignore_index=True)
    if "run_label" not in data:
        data["run_label"]="unspecified"
    data["run_label"]=data["run_label"].fillna("unspecified")
    for column in ("target_numeric_ks","nontarget_numeric_ks","target_categorical_tv","nontarget_categorical_tv"):
        if column not in data:
            data[column]=float("nan")
    data.to_csv(args.output/"all_metrics.csv",index=False)
    summary=data.groupby(["dataset","run_label","representation","method","guidance_scale","split"]).agg(
        seeds=("seed","nunique"),contracts=("contract_id","nunique"),satisfaction=("satisfaction_rate","mean"),
        cohort_ks=("cohort_numeric_ks","mean"),categorical_tv=("cohort_categorical_tv","mean"),
        target_numeric_ks=("target_numeric_ks","mean"),nontarget_numeric_ks=("nontarget_numeric_ks","mean"),
        target_categorical_tv=("target_categorical_tv","mean"),nontarget_categorical_tv=("nontarget_categorical_tv","mean"),
        edit_rate=("edit_rate","mean"),numeric_target_distortion=("numeric_target_distortion","mean"),
        categorical_target_edit_rate=("categorical_target_edit_rate","mean")).reset_index()
    summary.to_csv(args.output/"summary.csv",index=False)
    lines=["# Mixed-type open-contract adapter","","Atomic numeric/categorical training; held-out numeric parameters and unseen conjunctions.","",
           "| dataset | variant | representation | method | CFG | split | seeds | contracts | satisfaction | numeric KS | categorical TV | edit rate | numeric distortion | categorical edit |",
           "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.run_label} | {row.representation} | {row.method} | {row.guidance_scale:g} | {row.split} | {row.seeds} | {row.contracts} | {row.satisfaction:.4f} | {row.cohort_ks:.4f} | {row.categorical_tv:.4f} | {row.edit_rate:.4f} | {row.numeric_target_distortion:.4f} | {row.categorical_target_edit_rate:.4f} |")
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(f"loaded {len(paths)} runs",flush=True)

if __name__=="__main__": main()
