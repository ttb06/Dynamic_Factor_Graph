#!/usr/bin/env python3
"""Measure how row-wise active constraint graphs change from middle to final state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dynamic_graph_conditional_repaint import eligible_edges
from typed_dynamic_graph_guidance import edge_masks


def pair_stats(left: np.ndarray, right: np.ndarray, prefix: str) -> dict:
    both = left & right; union = left | right
    return {
        f"{prefix}_middle_pairs": int(left.sum()), f"{prefix}_final_pairs": int(right.sum()),
        f"{prefix}_persistent_pairs": int(both.sum()), f"{prefix}_middle_only_pairs": int((left & ~right).sum()),
        f"{prefix}_final_only_pairs": int((~left & right).sum()),
        f"{prefix}_jaccard": float(both.sum() / max(union.sum(), 1)),
        f"{prefix}_pair_flip_rate": float((left != right).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--max-calibration-rate", type=float, default=.05)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for dataset, graph_index in [("shoppers", 1), ("default", 0)]:
        graph_path = a.root / f"bootstrap_conditional_graph_{dataset}_1070443_{graph_index}" / "graph.json"
        edges = eligible_edges(json.loads(graph_path.read_text()), a.max_calibration_rate)
        for seed in [20261444, 20261445, 20261446]:
            runs = sorted(a.root.glob(f"decoupled_activation_global_{dataset}_s{seed}_1079600_*/samples"))
            if len(runs) != 1: raise RuntimeError(f"expected one run, got {runs}")
            middle = pd.read_csv(runs[0] / "trigger_preview.csv"); final = pd.read_csv(runs[0] / "tabdiff.csv")
            active_mid=[]; active_final=[]; bad_mid=[]; bad_final=[]
            for edge in edges:
                am, _, bm = edge_masks(middle, edge); af, _, bf = edge_masks(final, edge)
                active_mid.append(am); active_final.append(af); bad_mid.append(bm); bad_final.append(bf)
            active_mid=np.column_stack(active_mid); active_final=np.column_stack(active_final)
            bad_mid=np.column_stack(bad_mid); bad_final=np.column_stack(bad_final)
            row = {"dataset":dataset, "seed":seed, "rows":len(middle), "eligible_edges":len(edges)}
            row.update(pair_stats(active_mid, active_final, "active")); row.update(pair_stats(bad_mid, bad_final, "violated"))
            row["rows_with_active_set_change"] = float((active_mid != active_final).any(axis=1).mean())
            row["rows_with_violation_set_change"] = float((bad_mid != bad_final).any(axis=1).mean())
            rows.append(row)
    per_seed=pd.DataFrame(rows); per_seed.to_csv(a.output / "per_seed.csv", index=False)
    summary=per_seed.groupby("dataset").agg({column:["mean","std"] for column in per_seed.columns if column not in {"dataset","seed"}})
    summary.columns=[f"{left}_{right}" for left,right in summary.columns]; summary.reset_index().to_csv(a.output / "summary.csv",index=False)
    cols=["dataset","active_jaccard_mean","active_pair_flip_rate_mean","rows_with_active_set_change_mean",
          "violated_jaccard_mean","violated_pair_flip_rate_mean","rows_with_violation_set_change_mean",
          "violated_middle_only_pairs_mean","violated_final_only_pairs_mean"]
    show=summary.reset_index()[cols].copy()
    for c in cols[1:]: show[c]=show[c].map(lambda x:f"{x:.4f}")
    headers=list(show.columns); lines=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
    lines += ["| "+" | ".join(map(str,row))+" |" for row in show.itertuples(index=False,name=None)]
    text="# Active graph transition audit\n\n"+"\n".join(lines)+"\n"
    (a.output/"RESULTS.md").write_text(text); print(text)


if __name__ == "__main__": main()
