#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--glob",default="cross_actuation_*_seed*/contract_metrics.csv"); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 paths=sorted(a.input_root.glob(a.glob));
 if not paths: raise FileNotFoundError("no baseline metrics")
 d=pd.concat([pd.read_csv(x).assign(source=str(x.parent)) for x in paths],ignore_index=True); d.to_csv(a.output/"all_metrics.csv",index=False)
 s=d.groupby(["dataset","method","split"]).agg(seeds=("seed","nunique"),satisfaction=("satisfaction_rate","mean"),satisfaction_std=("satisfaction_rate","std"),cohort_ks=("cohort_numeric_ks","mean"),categorical_tv=("cohort_categorical_tv","mean"),edit_rate=("edit_rate","mean"),target_distortion=("target_distortion","mean"),passes=("passes","mean")).reset_index(); s.to_csv(a.output/"summary.csv",index=False)
 lines=["# Cross-dataset actuation baselines","","Proxy mechanisms on the same controlled cohort contracts.","","| dataset | method | split | seeds | satisfaction | cohort KS | categorical TV | edit rate | target distortion | passes |","| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for r in s.itertuples(index=False): lines.append(f"| {r.dataset} | {r.method} | {r.split} | {r.seeds} | {r.satisfaction:.4f} | {r.cohort_ks:.4f} | {r.categorical_tv:.4f} | {r.edit_rate:.4f} | {r.target_distortion:.4f} | {r.passes:g} |")
 (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print(f"loaded {len(paths)} runs")
if __name__=="__main__": main()
