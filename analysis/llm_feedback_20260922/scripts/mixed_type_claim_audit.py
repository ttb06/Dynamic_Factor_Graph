#!/usr/bin/env python3
"""Paired contract-seed bootstrap for mixed-type claims."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float,float,float]:
    values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]
    if not len(values): return np.nan,np.nan,np.nan
    indices=rng.integers(0,len(values),size=(draws,len(values)))
    means=values[indices].mean(axis=1)
    return float(values.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--condition-metrics",type=Path,required=True); parser.add_argument("--certifier-metrics",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--draws",type=int,default=30000); parser.add_argument("--seed",type=int,default=20260928)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False); rng=np.random.default_rng(args.seed); rows=[]
    data=pd.read_csv(args.condition_metrics)
    for (dataset,split), group in data[data.guidance_scale.eq(1)].groupby(["dataset","split"]):
        structured=group[(group.representation.eq("structured")) & (group.method.eq("structured_condition"))]
        request=group[(group.representation.eq("request_id")) & (group.method.eq("request_id_condition"))]
        paired=structured.merge(request,on=["dataset","split","contract_id","seed"],suffixes=("_structured","_request"))
        effects={
            "satisfaction_gain":paired.satisfaction_rate_structured-paired.satisfaction_rate_request,
            "numeric_ks_reduction":paired.cohort_numeric_ks_request-paired.cohort_numeric_ks_structured,
            "categorical_tv_reduction":paired.cohort_categorical_tv_request-paired.cohort_categorical_tv_structured,
        }
        pareto=(effects["satisfaction_gain"].ge(0)&effects["numeric_ks_reduction"].ge(0)&effects["categorical_tv_reduction"].ge(0))
        for metric,values in effects.items():
            mean,lo,hi=interval(values.to_numpy(),rng,args.draws); rows.append({"comparison":"structured_cfg1_vs_request_id_cfg1","dataset":dataset,"split":split,"metric":metric,"pairs":len(values),"mean":mean,"ci_low":lo,"ci_high":hi,"pareto_rate":float(pareto.mean())})
    cert=pd.read_csv(args.certifier_metrics)
    for (dataset,split), group in cert.groupby(["dataset","split"]):
        hybrid=group[(group.method.eq("structured_then_final_projection")) & (group.guidance_scale.eq(1))]
        baseline=group[(group.method.eq("unconditional_then_final_projection")) & (group.guidance_scale.eq(0))]
        paired=hybrid.merge(baseline,on=["dataset","split","contract_id","seed"],suffixes=("_hybrid","_baseline"))
        effects={
            "edit_rate_reduction":paired.edit_rate_baseline-paired.edit_rate_hybrid,
            "numeric_distortion_reduction":paired.numeric_target_distortion_baseline-paired.numeric_target_distortion_hybrid,
            "numeric_ks_reduction":paired.cohort_numeric_ks_baseline-paired.cohort_numeric_ks_hybrid,
            "categorical_tv_reduction":paired.cohort_categorical_tv_baseline-paired.cohort_categorical_tv_hybrid,
        }
        pareto=(effects["edit_rate_reduction"].ge(0)&effects["numeric_ks_reduction"].ge(0)&effects["categorical_tv_reduction"].ge(0))
        for metric,values in effects.items():
            mean,lo,hi=interval(values.to_numpy(),rng,args.draws); rows.append({"comparison":"hybrid_cfg1_vs_unconditional_same_certifier","dataset":dataset,"split":split,"metric":metric,"pairs":len(values),"mean":mean,"ci_low":lo,"ci_high":hi,"pareto_rate":float(pareto.mean())})
    result=pd.DataFrame(rows); result.to_csv(args.output/"paired_effects.csv",index=False)
    lines=["# Mixed-type paired claim audit","","Percentile bootstrap over matched contract–seed cells. Positive values favor structured/hybrid.","",
           "| comparison | dataset | split | metric | pairs | mean | 95% interval | Pareto rate |","| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in result.itertuples(index=False):
        lines.append(f"| {row.comparison} | {row.dataset} | {row.split} | {row.metric} | {row.pairs} | {row.mean:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] | {row.pareto_rate:.3f} |")
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    (args.output/"run_metadata.json").write_text(json.dumps({"status":"executed","draws":args.draws,"seed":args.seed,"pair_unit":"contract-seed; descriptive for fixed controlled catalog"},indent=2))
    print(result.to_string(index=False),flush=True)

if __name__=="__main__": main()
