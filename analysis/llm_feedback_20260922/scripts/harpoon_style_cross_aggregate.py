#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 paths=sorted(a.input_root.glob("harpoon_style_*_seed*/contract_metrics.csv"));
 if not paths: raise FileNotFoundError("no Harpoon-style metrics")
 d=pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in paths],ignore_index=True); d.to_csv(a.output/"all_metrics.csv",index=False)
 s=d.groupby(["dataset","method","eta","split"]).agg(seeds=("seed","nunique"),satisfaction=("satisfaction_rate","mean"),cohort_ks=("cohort_numeric_ks","mean"),categorical_tv=("cohort_categorical_tv","mean"),edit_rate=("edit_rate","mean"),target_distortion=("target_distortion","mean"),gradient_steps=("gradient_steps","mean"),runtime_seconds=("runtime_seconds","mean")).reset_index(); s.to_csv(a.output/"summary.csv",index=False)
 lines=["# Harpoon-style differentiable guidance proxy","","Not a faithful Harpoon reproduction; squared-hinge gradient on predicted-clean numeric lower bounds.","","| dataset | method | eta | split | seeds | satisfaction | KS | TV | edit | distortion | grad steps | seconds/contract |","| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
 for r in s.itertuples(index=False): lines.append(f"| {r.dataset} | {r.method} | {r.eta:g} | {r.split} | {r.seeds} | {r.satisfaction:.4f} | {r.cohort_ks:.4f} | {r.categorical_tv:.4f} | {r.edit_rate:.4f} | {r.target_distortion:.4f} | {r.gradient_steps:.1f} | {r.runtime_seconds:.3f} |")
 (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print(f"loaded {len(paths)} runs")
if __name__=="__main__": main()
