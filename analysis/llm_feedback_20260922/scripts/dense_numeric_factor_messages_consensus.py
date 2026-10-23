#!/usr/bin/env python3
"""Reference+witness action-space consensus driver on the ARR sampler."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import dense_numeric_factor_messages_arr as base
from batch_distribution_transport import ConditionalBatchDistributionTransport
from consensus_guidance_transport import ConsensusGuidanceTransport, VALID_MODES
from factorized_residual_witness import FactorizedResidualWitnessTransport
from posterior_quantile_transport import PosteriorQuantileConditionalBatchDistributionTransport


_CONSENSUS: ConsensusGuidanceTransport | None = None


def _value(tokens: list[str], name: str, default: str | None = None) -> str | None:
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


def _replace(tokens: list[str], name: str, value: str) -> list[str]:
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(tokens):
        token = tokens[index]
        if token == name:
            output.extend([name, value])
            index += 2
            replaced = True
        elif token.startswith(name + "="):
            output.append(f"{name}={value}")
            index += 1
            replaced = True
        else:
            output.append(token)
            index += 1
    if not replaced:
        output.extend([name, value])
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--consensus-mode", choices=["none", *sorted(VALID_MODES)], default="none")
    parser.add_argument("--consensus-witness-ratio", type=float, default=0.25)
    parser.add_argument("--consensus-witness-trees", type=int, default=32)
    parser.add_argument("--consensus-witness-min-r2", type=float, default=0.02)
    parser.add_argument("--consensus-witness-pair-heads", type=int, default=4)
    parser.add_argument("--consensus-witness-top-factors-per-row", type=int, default=1)
    parser.add_argument("--consensus-witness-top-rows-per-factor", type=float, default=0.15)
    parser.add_argument("--consensus-witness-min-score", type=float, default=0.25)
    parser.add_argument("--consensus-witness-clip", type=float, default=2.0)
    parser.add_argument("--consensus-seed", type=int, default=None)
    return parser


def _aggregate(diagnostics: dict[str, Any]) -> dict[str, Any]:
    trace = diagnostics.get("trace", [])
    steps = [step for step in trace if any(key.startswith("consensus_") for key in step)]
    diagnostics["consensus_calls"] = len(steps)
    if not steps:
        return diagnostics
    additive = {
        "consensus_reference_cell_events",
        "consensus_witness_cell_events",
        "consensus_extra_cell_events",
        "consensus_extra_row_events",
        "consensus_conflict_rows",
        "consensus_witness_selected_cells",
        "consensus_witness_selected_rows",
    }
    keys = sorted({key for step in steps for key in step if key.startswith("consensus_")})
    for key in keys:
        values = [float(step[key]) for step in steps if key in step]
        if values:
            diagnostics[key] = float(np.sum(values) if key in additive else np.mean(values))
    return diagnostics


def main() -> None:
    global _CONSENSUS
    parser = _parser()
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(parser.format_help())
        print("Base ARR options follow:\n")
    args, base_tokens = parser.parse_known_args(sys.argv[1:])
    if args.consensus_mode == "none":
        sys.argv = [sys.argv[0], *base_tokens]
        base.main()
        return
    if _value(base_tokens, "--distribution-transport-mode", "none") != "none":
        parser.error("active consensus requires --distribution-transport-mode none")
    strength = float(_value(base_tokens, "--distribution-transport-strength", "0"))
    if strength <= 0:
        parser.error("active consensus requires positive transport strength")
    repo_value = _value(base_tokens, "--repo")
    dataset = _value(base_tokens, "--dataset", "news")
    output_value = _value(base_tokens, "--output")
    run_seed = int(_value(base_tokens, "--seed", "20261601"))
    if repo_value is None or output_value is None:
        parser.error("--repo and --output remain required")
    info = json.loads((Path(repo_value) / "data" / str(dataset) / "info.json").read_text())
    target_indices = list(info.get("target_col_idx", []))
    if not target_indices:
        parser.error("consensus requires a task target")
    target_column = info["column_names"][target_indices[0]]
    seed = run_seed if args.consensus_seed is None else args.consensus_seed

    class _Factory:
        def __new__(cls, train, numeric_columns, edges, **kwargs):
            del cls, kwargs
            global _CONSENSUS
            if info["task_type"] == "regression":
                reference = PosteriorQuantileConditionalBatchDistributionTransport(
                    train,
                    numeric_columns,
                    edges,
                    condition_column=target_column,
                    quantile_bins=4,
                    temperature=0.75,
                    freeze_membership=False,
                    scope="all",
                    mode="copula",
                    seed=seed,
                )
            else:
                reference = ConditionalBatchDistributionTransport(
                    train,
                    numeric_columns,
                    edges,
                    condition_column=target_column,
                    scope="all",
                    mode="copula",
                    preserve_tail_fraction=0.10,
                    seed=seed,
                )
            witness = FactorizedResidualWitnessTransport(
                train,
                numeric_columns,
                edges,
                mode="marginal_pair",
                target_column=target_column,
                trees=args.consensus_witness_trees,
                min_r2=args.consensus_witness_min_r2,
                max_pair_heads=args.consensus_witness_pair_heads,
                top_factors_per_row=args.consensus_witness_top_factors_per_row,
                top_rows_per_factor=args.consensus_witness_top_rows_per_factor,
                min_score=args.consensus_witness_min_score,
                clip=args.consensus_witness_clip,
                seed=seed,
            )
            _CONSENSUS = ConsensusGuidanceTransport(
                train,
                reference,
                witness,
                mode=args.consensus_mode,
                witness_ratio=args.consensus_witness_ratio,
                target_column=target_column,
            )
            return _CONSENSUS

    original_sample = base.sample

    def _sample(*sample_args, **sample_kwargs):
        encoded, diagnostics = original_sample(*sample_args, **sample_kwargs)
        return encoded, _aggregate(diagnostics)

    base.BatchDistributionTransport = _Factory
    base.sample = _sample
    base_tokens = _replace(base_tokens, "--distribution-transport-mode", "marginal")
    sys.argv = [sys.argv[0], *base_tokens]
    base.main()

    output = Path(output_value)
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["distribution_transport_mode"] = "none"
    metadata["heldout_used_for_fitting"] = False
    metadata["consensus"] = _CONSENSUS.metadata() if _CONSENSUS is not None else None
    metadata["mechanism"] = (
        "locked conditional transport plus cross-fitted witness action-space gradient surgery"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    with (output / "RESULTS.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Action-space consensus\n\n"
            f"- Mode: `{args.consensus_mode}`\n"
            f"- Witness/reference norm ratio: `{args.consensus_witness_ratio}`\n"
            f"- Target follows locked reference: `{target_column}`\n"
        )


if __name__ == "__main__":
    main()
