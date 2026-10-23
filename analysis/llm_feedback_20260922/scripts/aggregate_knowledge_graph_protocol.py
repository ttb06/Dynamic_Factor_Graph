#!/usr/bin/env python3
"""Aggregate graph-generation protocol runs without reopening source data."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(path: Path):
    return json.loads(path.read_text())


def fmt(value, digits=4):
    return "—" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    comparisons, rejection_rows, claim_rows, examples = [], [], [], []
    for directory in args.inputs:
        metadata = load_json(directory / "run_metadata.json")
        dataset = metadata["dataset"]
        comparison = pd.read_csv(directory / "graph_comparison.csv")
        comparison.insert(0, "dataset", dataset)
        comparison["total_tokens"] = metadata.get("usage", {}).get("total_tokens", 0)
        comparison["api_calls"] = metadata.get("api_calls", 0)
        comparisons.append(comparison)

        graphs = {
            "strong_single_pass": load_json(directory / "strong_single_pass_graph.json"),
            "llm_protocol": load_json(directory / "llm_protocol_graph.json"),
            "empirical_only": load_json(directory / "empirical_graph.json"),
            "unified_protocol": load_json(directory / "unified_graph.json"),
        }
        canonical = {name: {item["canonical"] for item in values} for name, values in graphs.items()}
        protocol_unique = canonical["llm_protocol"] - canonical["strong_single_pass"]
        protocol_nonempirical = canonical["llm_protocol"] - canonical["empirical_only"]
        claim_rows.append({
            "dataset": dataset,
            "protocol_retained": len(graphs["llm_protocol"]),
            "single_pass_retained": len(graphs["strong_single_pass"]),
            "protocol_unique_vs_single": len(protocol_unique),
            "protocol_nonoverlap_empirical": len(protocol_nonempirical),
            "protocol_external_or_hybrid": sum(
                x.get("evidence_basis") in {"external_domain", "hybrid"} for x in graphs["llm_protocol"]
            ),
            "protocol_domain_general": sum(
                x.get("generalizability") == "domain_general" for x in graphs["llm_protocol"]
            ),
            "protocol_with_exception_groups": sum(bool(x.get("exceptions")) for x in graphs["llm_protocol"]),
            "retained_round1": sum(str(x.get("candidate_id", "")).startswith("r1_") for x in graphs["llm_protocol"]),
            "retained_round2": sum(str(x.get("candidate_id", "")).startswith("r2_") for x in graphs["llm_protocol"]),
            "retained_predefined_seed": sum(str(x.get("candidate_id", "")).startswith("seed_") for x in graphs["llm_protocol"]),
            "protocol_heldout_shift_gt_005": sum(
                float(x.get("heldout_shift", 0)) > .05 for x in graphs["llm_protocol"]
            ),
        })
        for item in graphs["llm_protocol"]:
            if item["canonical"] in protocol_unique:
                examples.append({
                    "dataset": dataset,
                    "description": item["description"],
                    "class": item["classification"],
                    "family": item["relation_family"],
                    "evidence": item["evidence_basis"],
                    "generalizability": item.get("generalizability"),
                    "calibration_support": item["calibration"]["support"],
                    "calibration_violation_rate": item["calibration"]["violation_rate"],
                    "heldout_support": item["heldout"]["support"],
                    "heldout_violation_rate": item["heldout"]["violation_rate"],
                    "heldout_shift": item.get("heldout_shift"),
                })

        rejections = load_json(directory / "stage_rejections.json")
        for stage, rows in rejections.items():
            reasons = Counter(
                row.get("rejection", row.get("selection_rejection", "unspecified")) for row in rows
            )
            for reason, count in reasons.items():
                rejection_rows.append({"dataset": dataset, "stage": stage, "reason": reason, "count": count})

    all_comparisons = pd.concat(comparisons, ignore_index=True)
    claims = pd.DataFrame(claim_rows)
    rejection_frame = pd.DataFrame(rejection_rows)
    example_frame = pd.DataFrame(examples)
    all_comparisons.to_csv(args.output / "all_graph_comparison.csv", index=False)
    claims.to_csv(args.output / "protocol_claim_audit.csv", index=False)
    rejection_frame.to_csv(args.output / "stage_rejection_counts.csv", index=False)
    example_frame.to_csv(args.output / "protocol_unique_examples.csv", index=False)

    lines = [
        "# Knowledge-graph generation protocol: aggregate audit",
        "",
        "## Graph yield and held-out stability",
        "",
        "`Retained/proposed` is a protocol yield, not precision against a semantic gold standard.",
        "",
        "| Dataset | Graph | Proposed | Retained | Hard | Near-hard | Soft | Families | Endpoint patterns | External/hybrid | Heldout stable ≤.05 | Mean shift |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in all_comparisons.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.graph} | {row.proposed} | {row.retained} | {row.hard} | "
            f"{row.near_hard} | {row.soft} | {row.family_diversity} | {row.endpoint_patterns} | "
            f"{row.external_or_hybrid} | {fmt(row.heldout_stable_005, 3)} | {fmt(row.mean_heldout_shift)} |"
        )
    lines += [
        "",
        "## Incremental value of the multi-stage protocol",
        "",
        "| Dataset | Protocol retained | Single-pass retained | Unique vs single-pass | Not in empirical graph | External/hybrid | Domain-general | Exceptions | Round 1 | Round 2 | Seeds | Shift >.05 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in claims.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.protocol_retained} | {row.single_pass_retained} | "
            f"{row.protocol_unique_vs_single} | {row.protocol_nonoverlap_empirical} | "
            f"{row.protocol_external_or_hybrid} | {row.protocol_domain_general} | "
            f"{row.protocol_with_exception_groups} | {row.retained_round1} | {row.retained_round2} | "
            f"{row.retained_predefined_seed} | {row.protocol_heldout_shift_gt_005} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `strong_single_pass` is the same stronger external model with the same schema and aggregate evidence, but without role separation, counterexample-guided refinement, critic, and independent judge.",
        "- `empirical_only` measures what discovery-split statistics can supply without foundation-model knowledge.",
        "- `llm_protocol` measures semantically reviewed foundation-model edges; `unified_protocol` merges these with empirical edges.",
        "- Calibration determines empirical hardness; it does not prove an invariant. Heldout is opened only after the graph is frozen.",
        "- A useful protocol should add stable, semantically defensible edges beyond both single-pass prompting and empirical mining. Retaining more edges alone is not success.",
        "",
        "## Privacy and leakage checks",
        "",
        "- API payloads contain schema and aggregate counts only; no source row is sent.",
        "- Empirical candidates are re-mined from the discovery partition, rather than imported from a catalog fitted on the full train split.",
        "- Counterexample data is used by the critic, blind calibration by deterministic selection, and heldout only for the final audit.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
