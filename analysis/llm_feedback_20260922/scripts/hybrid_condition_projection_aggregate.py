#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 paths=sorted(a.input_root.glob("hybrid_projection_*_seed*/contract_metrics.csv"));
 if not paths: raise FileNotFoundError("no hybrid metrics")
 d=pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in paths],ignore_index=True); d.to_csv(a.output/"all_metrics.csv",index=False)
 s=d.groupby(["dataset","method","guidance_scale","split"]).agg(seeds=("seed","nunique"),satisfaction=("satisfaction_rate","mean"),cohort_ks=("cohort_numeric_ks","mean"),categorical_tv=("cohort_categorical_tv","mean"),edit_rate=("edit_rate","mean"),target_distortion=("target_distortion","mean")).reset_index(); s.to_csv(a.output/"summary.csv",index=False)
 lines=["# Structured condition followed by exact final projection","","Evaluation-only, 4,096 samples per held-out contract. Projection uses the same typed numeric contract.","","| dataset | method | CFG | split | seeds | satisfaction | cohort KS | categorical TV | edit rate | target distortion |","| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for r in s.itertuples(index=False): lines.append(f"| {r.dataset} | {r.method} | {r.guidance_scale:g} | {r.split} | {r.seeds} | {r.satisfaction:.4f} | {r.cohort_ks:.4f} | {r.categorical_tv:.4f} | {r.edit_rate:.4f} | {r.target_distortion:.4f} |")
 (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print(f"loaded {len(paths)} runs")
if __name__=="__main__": main()
