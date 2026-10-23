#!/usr/bin/env python3
"""Aggregate soft/shuffled/oracle applicability controls."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    paths=sorted(a.input_root.glob("prob_gate_g*_t*_seed*/control_metrics.csv"))
    if not paths: raise FileNotFoundError("no control metrics")
    data=pd.concat([pd.read_csv(x).assign(source=str(x.parent)) for x in paths],ignore_index=True)
    data.to_csv(a.output/"all_metrics.csv",index=False)
    s=data.groupby(["guidance_scale","gate_start_t","schedule","early_policy","early_floor","split"]).agg(
      seeds=("seed","nunique"),violation=("conditional_violation_rate","mean"),violation_std=("conditional_violation_rate","std"),
      reduction=("conditional_violation_reduction","mean"),antecedent_error=("antecedent_rate_error","mean"),
      exception_error=("exception_rate_error","mean"),inactive_error=("inactive_target_mean_error","mean"),
      numeric_ks=("global_numeric_ks","mean"),categorical_tv=("global_categorical_tv","mean")).reset_index()
    s.to_csv(a.output/"summary.csv",index=False)
    lines=["# Probabilistic applicability and causal controls","","All conditional deltas are target-scoped and use paired initial noise.",""]
    lines += ["| CFG | gate start | schedule | split | seeds | violation | reduction | antecedent error | exception error | inactive error | numeric KS | categorical TV |",
              "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in s.itertuples(index=False):
      lines.append(f"| {r.guidance_scale:g} | {r.gate_start_t:g} | {r.schedule} | {r.split} | {r.seeds} | {r.violation:.4f} | {r.reduction:.4f} | {r.antecedent_error:.4f} | {r.exception_error:.4f} | {r.inactive_error:.4f} | {r.numeric_ks:.4f} | {r.categorical_tv:.4f} |")
    (a.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print(f"loaded {len(paths)} runs -> {a.output}")

if __name__=="__main__": main()
