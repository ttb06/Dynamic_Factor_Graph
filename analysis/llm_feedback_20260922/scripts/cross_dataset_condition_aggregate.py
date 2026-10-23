#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 paths=sorted(a.input_root.glob("cross_condition_*_seed*/contract_metrics.csv"));
 if not paths: raise FileNotFoundError("no cross condition runs")
 d=pd.concat([pd.read_csv(x).assign(source=str(x.parent)) for x in paths],ignore_index=True); d.to_csv(a.output/"all_metrics.csv",index=False)
 s=d.groupby(["dataset","representation","guidance_scale","split"]).agg(seeds=("seed","nunique"),satisfaction=("satisfaction_rate","mean"),satisfaction_std=("satisfaction_rate","std"),gain_null=("gain_null","mean"),gain_frozen=("gain_frozen","mean"),cohort_ks=("cohort_numeric_ks","mean"),categorical_tv=("cohort_categorical_tv","mean")).reset_index(); s.to_csv(a.output/"summary.csv",index=False)
 lines=["# Cross-dataset controlled contract transfer","","Contracts are controlled cohort requests, not claimed domain invariants.","","| dataset | representation | CFG | split | seeds | satisfaction | gain over null | cohort KS | categorical TV |","| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
 for r in s.itertuples(index=False): lines.append(f"| {r.dataset} | {r.representation} | {r.guidance_scale:g} | {r.split} | {r.seeds} | {r.satisfaction:.4f} | {r.gain_null:.4f} | {r.cohort_ks:.4f} | {r.categorical_tv:.4f} |")
 (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print(f"loaded {len(paths)} runs")
if __name__=="__main__": main()
