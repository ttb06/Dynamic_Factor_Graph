#!/usr/bin/env python3
"""Promote a score-only edge only when two independent FM reviews agree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--graph",type=Path,required=True)
    parser.add_argument("--review-a",type=Path,required=True)
    parser.add_argument("--review-b",type=Path,required=True)
    parser.add_argument("--min-confidence",type=float,default=.80)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    graph=json.loads(args.graph.read_text())
    reviews=[]
    for path in (args.review_a,args.review_b):
        items=json.loads((path/"semantic_decisions.json").read_text())
        reviews.append({item["candidate_id"]:item for item in items})
    semantic={"domain","logical","process","accounting","temporal"}
    output=[]; decisions=[]; promoted=[]
    for edge in graph:
        item=dict(edge); pair=[review.get(edge["candidate_id"]) for review in reviews]
        eligible=(edge.get("usage_role")=="score_only" and all(pair)
                  and all(d["safe_to_actuate"] and d["confidence"]>=args.min_confidence
                          and d["semantic_kind"] in semantic and d["preferred_action"]!="abstain" for d in pair))
        if all(pair):
            item["semantic_promotion_consensus"]={"reviews":pair,"agreed_safe":eligible}
            decisions.append({"candidate_id":edge["candidate_id"],"agreed_safe":eligible,
                              "confidence_a":pair[0]["confidence"],"confidence_b":pair[1]["confidence"],
                              "kind_a":pair[0]["semantic_kind"],"kind_b":pair[1]["semantic_kind"]})
        if eligible:
            item["usage_role"]="actuate"; item["selection_weight"]=100.0
            item["role_rationale"]="two-FM semantic consensus plus local aggregate calibration approved actuation"
            item["promotion_source"]="two_fm_semantic_consensus"
            item["judge_confidence"]=min(pair[0]["confidence"],pair[1]["confidence"])
            promoted.append(item["candidate_id"])
        output.append(item)
    (args.output/"constraint_catalog.json").write_text(json.dumps(output,indent=2))
    (args.output/"consensus_decisions.json").write_text(json.dumps(decisions,indent=2))
    metadata={"promoted_ids":promoted,"promoted_count":len(promoted),"min_confidence":args.min_confidence,
              "review_a":str(args.review_a),"review_b":str(args.review_b),"heldout_used":False}
    (args.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
    lines=["# Two-model semantic promotion consensus","",f"Promoted: `{len(promoted)}`: `{promoted}`.","",
           "| ID | Agreed safe | Confidence A | Confidence B | Kind A | Kind B |",
           "| --- | --- | ---: | ---: | --- | --- |"]
    for d in decisions:
        lines.append(f"| {d['candidate_id']} | {d['agreed_safe']} | {d['confidence_a']:.3f} | {d['confidence_b']:.3f} | {d['kind_a']} | {d['kind_b']} |")
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print("\n".join(lines))


if __name__=="__main__":
    main()
