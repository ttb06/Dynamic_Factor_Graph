#!/usr/bin/env python3
"""Aggregate large-sample cross-dataset condition evaluations."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--input-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    paths=sorted(args.input_root.glob("cross_large_eval_*_seed*/contract_metrics.csv"))
    if not paths: raise FileNotFoundError("no large evaluation metrics")
    data=pd.concat([pd.read_csv(path).assign(source=str(path.parent)) for path in paths],ignore_index=True)
    data.to_csv(args.output/"all_metrics.csv",index=False)
    summary=data.groupby(["dataset","guidance_scale","split"]).agg(
        seeds=("seed","nunique"),satisfaction=("satisfaction_rate","mean"),satisfaction_std=("satisfaction_rate","std"),
        gain_null=("gain_null","mean"),cohort_ks=("cohort_numeric_ks","mean"),categorical_tv=("cohort_categorical_tv","mean")
    ).reset_index(); summary.to_csv(args.output/"summary.csv",index=False)
    lines=["# Large-sample cross-dataset evaluation","","Evaluation-only: 4,096 samples per held-out contract; trained adapters are unchanged.","",
           "| dataset | CFG | split | seeds | satisfaction | gain over null | cohort KS | categorical TV |",
           "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.guidance_scale:g} | {row.split} | {row.seeds} | {row.satisfaction:.4f} | {row.gain_null:.4f} | {row.cohort_ks:.4f} | {row.categorical_tv:.4f} |")
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print(f"loaded {len(paths)} runs -> {args.output}")


if __name__=="__main__": main()
