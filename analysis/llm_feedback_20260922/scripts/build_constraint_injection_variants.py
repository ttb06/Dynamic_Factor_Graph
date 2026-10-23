#!/usr/bin/env python3
"""Build graph variants that separate semantic authority from soft messages.

Documented edges may actuate.  Soft edges only define adjacency for a
train-fitted conditional factor; their projection weight is forced to zero.
The variants isolate whether LLM filtering of statistical relations improves
the usefulness of those soft factor messages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--luna", type=Path, required=True)
    parser.add_argument("--terra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    luna = json.loads(args.luna.read_text()); terra = json.loads(args.terra.read_text())
    terra_by_program = {
        json.dumps(item["program"], sort_keys=True): item for item in terra
    }

    documented, consensus_soft, luna_soft, stable_non_documented = [], [], [], []
    audit = []
    for item in luna:
        key = json.dumps(item["program"], sort_keys=True)
        peer = terra_by_program.get(key)
        luna_decision = item.get("semantic_review", {}).get("decision")
        terra_decision = peer.get("semantic_review", {}).get("decision") if peer else "missing"
        edge = dict(item)
        if item.get("documented_match"):
            edge["guidance_weight"] = 1.0
            edge["usage_role"] = "actuate"
            documented.append(edge)
            bucket = "documented_actuate"
        else:
            # Soft relations are never passed to the projection layer.  They
            # can only connect endpoints in DenseNumericFactors, whose model
            # and reliability are fitted on train with a validation R2 gate.
            edge["guidance_weight"] = 0.0
            edge["usage_role"] = "soft_factor_only"
            stable_non_documented.append(edge)
            if luna_decision == "semantic_soft_score": luna_soft.append(edge)
            if luna_decision == terra_decision == "semantic_soft_score":
                consensus_soft.append(edge)
            bucket = "non_documented"
        audit.append({"candidate_id": item["candidate_id"], "description": item["description"],
                      "luna": luna_decision, "terra": terra_decision, "bucket": bucket})

    variants = {
        "documented_only": documented,
        "documented_plus_consensus_soft": documented + consensus_soft,
        "documented_plus_luna_soft": documented + luna_soft,
        "documented_plus_unfiltered_stable": documented + stable_non_documented,
    }
    order_actuation = [
        edge for edge in documented if edge.get("program", {}).get("kind") == "ordered_triplet"
    ]
    variants.update({
        "order_actuation_only": order_actuation,
        "order_plus_consensus_soft": order_actuation + consensus_soft,
        "order_plus_luna_soft": order_actuation + luna_soft,
        "order_plus_unfiltered_stable": order_actuation + stable_non_documented,
    })
    for name, edges in variants.items():
        (args.output/f"{name}.json").write_text(json.dumps(edges, indent=2))
    summary = {name: {"edges": len(edges),
                      "actuate": sum(edge["usage_role"] == "actuate" for edge in edges),
                      "soft_factor_only": sum(edge["usage_role"] == "soft_factor_only" for edge in edges)}
               for name, edges in variants.items()}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    (args.output/"semantic_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
