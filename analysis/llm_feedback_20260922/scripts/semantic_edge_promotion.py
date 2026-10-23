#!/usr/bin/env python3
"""Use an FM to decide which calibrated empirical edges are safe to actuate.

The API receives no real rows and no heldout statistics.  It sees schema-level
edge programs plus aggregate discovery/calibration evidence, and must separate
domain/process semantics from dataset-specific correlations.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from llm_feedback_audit import load_env
from semantic_llm_sweep import api_json, output_text


KINDS=["domain","logical","process","accounting","temporal","distributional_correlation","uncertain"]
ACTIONS=["edit_consequent","deactivate_antecedent","choose_minimal_counterfactual","abstain"]
OUTPUT_SCHEMA={
    "type":"object","properties":{"decisions":{"type":"array","items":{
        "type":"object","properties":{
            "candidate_id":{"type":"string"},
            "safe_to_actuate":{"type":"boolean"},
            "semantic_kind":{"type":"string","enum":KINDS},
            "confidence":{"type":"number","minimum":0,"maximum":1},
            "preferred_action":{"type":"string","enum":ACTIONS},
            "plausible_exception":{"type":"string"},
            "reason":{"type":"string"},
        },"required":["candidate_id","safe_to_actuate","semantic_kind","confidence",
                       "preferred_action","plausible_exception","reason"],
        "additionalProperties":False}},
    },"required":["decisions"],"additionalProperties":False,
}


def compact(edge: dict) -> dict:
    keep={"candidate_id":edge["candidate_id"],"description":edge.get("description",""),
          "applicability":edge.get("applicability",[]),"requirements":edge.get("requirements",[]),
          "discovery":edge.get("discovery",{}),"counterexample":edge.get("counterexample",{}),
          "calibration":edge.get("calibration",{}),"classification":edge.get("classification"),
          "source":edge.get("source"),"evidence_basis":edge.get("evidence_basis")}
    # Heldout evidence is deliberately excluded from model selection.
    for split in ("discovery","counterexample","calibration"):
        keep[split]={key:value for key,value in keep[split].items()
                     if key in {"support","violations","violation_rate","base_rate","lift","coverage"}}
    return keep


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--graph",type=Path,required=True)
    parser.add_argument("--env-file",type=Path,required=True)
    parser.add_argument("--model",default="gpt-5.4")
    parser.add_argument("--min-confidence",type=float,default=.80)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    load_env(args.env_file); key=os.environ.get("OPENAI_API_KEY")
    if not key: raise RuntimeError("OPENAI_API_KEY unavailable")
    graph=json.loads(args.graph.read_text())
    candidates=[edge for edge in graph if edge.get("usage_role")=="score_only"]
    payload={"dataset":args.dataset,"privacy":"schema programs and aggregates only; no rows",
             "candidates":[compact(edge) for edge in candidates],
             "task":"Decide whether each empirical implication is a defensible semantic constraint for selective generation-time actuation."}
    instructions=(
        "You are the semantic safety judge in a synthetic-tabular constraint pipeline. Review every candidate ID. "
        "Mark safe_to_actuate true only when the implication follows from domain, logical, temporal, accounting, "
        "or process semantics and a generated row violating it would normally be considered invalid. High empirical "
        "confidence, lift, common co-occurrence, or dataset coding alone are not sufficient. Reject browser/traffic, "
        "month/season, demographic, target, and other correlations unless a genuine invariant is explicit. Near-hard "
        "rules may be safe only with calibrated exceptions; describe a plausible exception. Prefer abstain whenever "
        "semantics are ambiguous. Do not infer facts from heldout data because none is supplied. Return one decision "
        "for every candidate exactly once."
    )
    body={"model":args.model,"instructions":instructions,"input":json.dumps(payload,ensure_ascii=False),
          "text":{"format":{"type":"json_schema","name":"semantic_edge_promotion","strict":True,
                            "schema":OUTPUT_SCHEMA}},"max_output_tokens":6000,"store":False}
    started=time.perf_counter(); response=api_json("https://api.openai.com/v1/responses",key,body,timeout=300)
    parsed=json.loads(output_text(response)); decisions={item["candidate_id"]:item for item in parsed["decisions"]}
    missing=sorted({edge["candidate_id"] for edge in candidates}-set(decisions))
    extra=sorted(set(decisions)-{edge["candidate_id"] for edge in candidates})
    if missing or extra: raise RuntimeError(f"decision coverage mismatch missing={missing} extra={extra}")

    promoted=[]; output_graph=[]
    allowed_kinds={"domain","logical","process","accounting","temporal"}
    for edge in graph:
        item=dict(edge); decision=decisions.get(edge["candidate_id"])
        if decision:
            item["semantic_promotion_review"]=decision
            eligible=(decision["safe_to_actuate"] and decision["confidence"]>=args.min_confidence
                      and decision["semantic_kind"] in allowed_kinds
                      and decision["preferred_action"]!="abstain")
            if eligible:
                item["usage_role"]="actuate"; item["selection_weight"]=100.0
                item["role_rationale"]="FM semantic review plus local aggregate calibration approved selective actuation"
                item["promotion_source"]="fm_semantic_review"
                item["judge_confidence"]=decision["confidence"]
                promoted.append(item["candidate_id"])
        output_graph.append(item)
    (args.output/"constraint_catalog.json").write_text(json.dumps(output_graph,indent=2))
    (args.output/"semantic_decisions.json").write_text(json.dumps(parsed["decisions"],indent=2))
    metadata={"dataset":args.dataset,"model":args.model,"candidate_count":len(candidates),
              "promoted_count":len(promoted),"promoted_ids":promoted,"min_confidence":args.min_confidence,
              "heldout_used":False,"raw_rows_sent":False,"usage":response.get("usage",{}),
              "latency_seconds":time.perf_counter()-started}
    (args.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
    lines=["# Semantic edge promotion","",f"Dataset: `{args.dataset}`. Model: `{args.model}`.",
           f"Promoted: `{len(promoted)}/{len(candidates)}`. Heldout used: `False`. Raw rows sent: `False`.","",
           "| ID | Safe | Kind | Confidence | Action | Reason |","| --- | --- | --- | ---: | --- | --- |"]
    for edge in candidates:
        d=decisions[edge["candidate_id"]]
        lines.append(f"| {edge['candidate_id']} | {d['safe_to_actuate']} | {d['semantic_kind']} | {d['confidence']:.3f} | {d['preferred_action']} | {d['reason'].replace('|','/')} |")
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print("\n".join(lines))


if __name__=="__main__":
    main()
