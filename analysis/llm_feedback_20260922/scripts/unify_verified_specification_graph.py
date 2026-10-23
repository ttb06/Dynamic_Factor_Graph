#!/usr/bin/env python3
"""Merge empirical, foundation-model and optional expert edges into one graph.

All sources are compiled to the same typed implication DSL and re-verified on
the same discovery/calibration split.  Provenance is retained; heldout data is
attached only after selection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from general_state_graph_probe import parse_implication, PREDICATE
from llm_verified_graph_discovery import verify


def predicate(text: str) -> dict:
    match = PREDICATE.match(text)
    if not match: raise ValueError(text)
    column, operator, value = (part.strip() for part in match.groups())
    try: value = float(value)
    except ValueError: pass
    return {"column": column, "operator": operator, "value": value}


def canonical(a: dict, c: dict) -> str:
    return json.dumps({"antecedent": a, "consequent": c}, sort_keys=True)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--dataset", required=True); p.add_argument("--empirical", type=Path, required=True)
    p.add_argument("--llm-dir", type=Path); p.add_argument("--expert-json", type=Path)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--min-support", type=int, default=100)
    p.add_argument("--max-edges", type=int, default=32); p.add_argument("--seed", type=int, default=20261004)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.repo/"analysis/llm_feedback_20260922/scripts"))
    import cross_dataset_constraint_audit as audit
    train, heldout, _, _, target = audit.load_dataset(a.repo, a.dataset)
    rng = np.random.default_rng(a.seed); order = rng.permutation(len(train)); cut = int(.7*len(train))
    discovery, calibration = train.iloc[order[:cut]], train.iloc[order[cut:]]
    proposals = []
    empirical = pd.read_csv(a.empirical)
    empirical = empirical[(empirical.dataset.eq(a.dataset)) & empirical.source.eq("train_only_predicate_mining")]
    for row in empirical.itertuples(index=False):
        try:
            left, right = parse_implication(row.constraint)
            proposals.append({"antecedent": predicate(left), "consequent": predicate(right),
                              "description": row.constraint, "source": "empirical_train_mining"})
        except Exception: pass
    if a.llm_dir and (a.llm_dir/"verified_graph.json").exists():
        for item in json.loads((a.llm_dir/"verified_graph.json").read_text()):
            proposals.append({"antecedent": item["antecedent"], "consequent": item["consequent"],
                              "description": item["description"], "source": "foundation_model_verified"})
    if a.expert_json and a.expert_json.exists():
        for item in json.loads(a.expert_json.read_text()):
            proposals.append({**item, "source": "expert_rule"})
    merged = {}
    for item in proposals:
        key = canonical(item["antecedent"], item["consequent"])
        if key not in merged: merged[key] = {**item, "provenance": [item["source"]]}
        elif item["source"] not in merged[key]["provenance"]: merged[key]["provenance"].append(item["source"])
    retained, rejected = [], []
    for key, item in merged.items():
        columns = [item["antecedent"]["column"], item["consequent"]["column"]]
        try:
            d, c, h = verify(discovery, item), verify(calibration, item), verify(heldout, item)
        except Exception as exc:
            rejected.append({**item, "reason": f"execution_error:{exc}"}); continue
        row = {**item, "canonical_edge": key, "discovery": d, "calibration": c, "heldout": h}
        if columns[0] == columns[1]: reason = "same_column"
        elif target in columns: reason = "target_related"
        elif audit.is_sensitive(columns): reason = "sensitive_related"
        elif min(d["support"], c["support"]) < a.min_support: reason = "low_support"
        elif max(d["base_rate"], c["base_rate"]) >= .995: reason = "vacuous_consequent"
        else: reason = ""
        if reason:
            row["reason"] = reason; rejected.append(row); continue
        worst_v = max(d["violation_rate"], c["violation_rate"])
        min_conf, min_lift = min(d["confidence"], c["confidence"]), min(d["lift"], c["lift"])
        if worst_v <= .01 and min_lift >= 1.01: hardness = "hard"
        elif worst_v <= .10 and min_conf >= .90 and min_lift >= 1.02: hardness = "near_hard"
        elif min_conf >= .70 and min_lift >= 1.20: hardness = "soft"
        else:
            row["reason"] = "weak_or_unstable"; rejected.append(row); continue
        row["hardness"] = hardness
        row["actuation_eligible"] = hardness in {"hard", "near_hard"}
        row["score"] = math.log1p(min(d["support"], c["support"])) * min(min_lift, 6) * (1-worst_v)
        retained.append(row)
    retained.sort(key=lambda x: -x["score"])
    # Limit repeated endpoints while retaining multiple knowledge sources.
    selected, pairs = [], {}
    for row in retained:
        pair = (row["antecedent"]["column"], row["consequent"]["column"])
        if pairs.get(pair, 0) >= 2: continue
        selected.append(row); pairs[pair] = pairs.get(pair, 0)+1
        if len(selected) >= a.max_edges: break
    (a.output/"unified_graph.json").write_text(json.dumps(selected, indent=2))
    (a.output/"all_retained.json").write_text(json.dumps(retained, indent=2))
    (a.output/"rejected.json").write_text(json.dumps(rejected, indent=2))
    flat = []
    for i, row in enumerate(selected):
        flat.append({"edge_index": i, "description": row["description"],
                     "antecedent": json.dumps(row["antecedent"], sort_keys=True),
                     "consequent": json.dumps(row["consequent"], sort_keys=True),
                     "provenance": "+".join(sorted(row["provenance"])), "hardness": row["hardness"],
                     "actuation_eligible": row["actuation_eligible"], "score": row["score"],
                     **{f"discovery_{k}": v for k,v in row["discovery"].items()},
                     **{f"calibration_{k}": v for k,v in row["calibration"].items()},
                     **{f"heldout_{k}": v for k,v in row["heldout"].items()}})
    pd.DataFrame(flat).to_csv(a.output/"unified_graph.csv", index=False)
    counts = pd.Series([r["hardness"] for r in selected]).value_counts().to_dict()
    provenance = pd.Series([p for r in selected for p in r["provenance"]]).value_counts().to_dict()
    summary = {"dataset": a.dataset, "raw_proposals": len(proposals), "unique_edges": len(merged),
               "retained_before_diversity": len(retained), "selected": len(selected),
               "hardness": counts, "provenance": provenance, "heldout_used_for_selection": False}
    (a.output/"summary.json").write_text(json.dumps(summary, indent=2))
    (a.output/"RESULTS.md").write_text("# Unified verified specification graph\n\n" +
        "```json\n" + json.dumps(summary, indent=2) + "\n```\n")


if __name__ == "__main__": main()
