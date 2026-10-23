#!/usr/bin/env python3
"""Factorized RFF-MMD trajectory driver layered on the verified ARR sampler."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import dense_numeric_factor_messages_arr as base
from factorized_rff_mmd import FactorizedRFFMMDTransport, VALID_MODES


_OPERATOR: FactorizedRFFMMDTransport | None = None


def _option_value(tokens: list[str], name: str, default: str | None = None) -> str | None:
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


def _replace_option(tokens: list[str], name: str, value: str) -> list[str]:
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(tokens):
        token = tokens[index]
        if token == name:
            output.extend([name, value])
            index += 2
            replaced = True
            continue
        if token.startswith(name + "="):
            output.append(f"{name}={value}")
            index += 1
            replaced = True
            continue
        output.append(token)
        index += 1
    if not replaced:
        output.extend([name, value])
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--factor-mmd-mode", choices=sorted(VALID_MODES), default="none"
    )
    parser.add_argument("--factor-mmd-features", type=int, default=64)
    parser.add_argument("--factor-mmd-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--factor-mmd-max-pair-heads", type=int, default=8)
    parser.add_argument("--factor-mmd-top-factors-per-row", type=int, default=1)
    parser.add_argument("--factor-mmd-top-rows-per-factor", type=float, default=0.15)
    parser.add_argument("--factor-mmd-min-score", type=float, default=0.25)
    parser.add_argument("--factor-mmd-action-clip", type=float, default=0.50)
    parser.add_argument("--factor-mmd-seed", type=int, default=None)
    return parser


def _aggregate(diagnostics: dict[str, Any]) -> dict[str, Any]:
    trace = diagnostics.get("trace", [])
    steps = [step for step in trace if any(key.startswith("factor_mmd_") for key in step)]
    diagnostics["factor_mmd_calls"] = len(steps)
    if not steps:
        return diagnostics
    additive = {
        "factor_mmd_candidate_row_factors",
        "factor_mmd_factor_budget_row_factors",
        "factor_mmd_row_budget_row_factors",
        "factor_mmd_selected_row_factors",
        "factor_mmd_selected_rows",
        "factor_mmd_selected_cells",
        "factor_mmd_conflicting_cells",
        "factor_mmd_marginal_selected_row_factors",
        "factor_mmd_pair_selected_row_factors",
        "factor_mmd_predicted_energy_decrease",
    }
    keys = sorted({
        key for step in steps for key in step if key.startswith("factor_mmd_")
    })
    for key in keys:
        values = [float(step[key]) for step in steps if key in step]
        if values:
            diagnostics[key] = float(np.sum(values) if key in additive else np.mean(values))
    return diagnostics


def main() -> None:
    global _OPERATOR
    parser = _parser()
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(parser.format_help())
        print("Base ARR options follow:\n")
    mmd_args, base_tokens = parser.parse_known_args(sys.argv[1:])
    if mmd_args.factor_mmd_mode == "none":
        sys.argv = [sys.argv[0], *base_tokens]
        base.main()
        return

    if _option_value(base_tokens, "--distribution-transport-mode", "none") != "none":
        parser.error("active factor-MMD requires --distribution-transport-mode none")
    incompatible = [
        token for token in base_tokens
        if token.startswith("--distribution-transport-condition-target")
        or token.startswith("--distribution-transport-soft-quantiles")
        or token.startswith("--distribution-transport-adaptive-resolution")
    ]
    if incompatible:
        parser.error("factor-MMD cannot be combined with conditional/adaptive transport")
    strength = float(_option_value(
        base_tokens, "--distribution-transport-strength", "0"
    ))
    if strength <= 0:
        parser.error("active factor-MMD requires positive transport strength")

    repo_value = _option_value(base_tokens, "--repo")
    dataset = _option_value(base_tokens, "--dataset", "news")
    output_value = _option_value(base_tokens, "--output")
    run_seed = int(_option_value(base_tokens, "--seed", "20261601"))
    if repo_value is None or output_value is None:
        parser.error("--repo and --output remain required base options")
    info = json.loads((Path(repo_value) / "data" / str(dataset) / "info.json").read_text())
    target_indices = list(info.get("target_col_idx", []))
    target_column = info["column_names"][target_indices[0]] if target_indices else None
    operator_seed = run_seed if mmd_args.factor_mmd_seed is None else mmd_args.factor_mmd_seed

    class _Factory:
        def __new__(cls, train, numeric_columns, edges, **kwargs):
            del cls, kwargs
            global _OPERATOR
            _OPERATOR = FactorizedRFFMMDTransport(
                train,
                numeric_columns,
                edges,
                mode=mmd_args.factor_mmd_mode,
                target_column=target_column,
                features=mmd_args.factor_mmd_features,
                bandwidth_scale=mmd_args.factor_mmd_bandwidth_scale,
                max_pair_heads=mmd_args.factor_mmd_max_pair_heads,
                top_factors_per_row=mmd_args.factor_mmd_top_factors_per_row,
                top_rows_per_factor=mmd_args.factor_mmd_top_rows_per_factor,
                min_score=mmd_args.factor_mmd_min_score,
                action_clip=mmd_args.factor_mmd_action_clip,
                seed=operator_seed,
            )
            return _OPERATOR

    original_sample = base.sample

    def _sample(*args, **kwargs):
        encoded, diagnostics = original_sample(*args, **kwargs)
        return encoded, _aggregate(diagnostics)

    base.BatchDistributionTransport = _Factory
    base.sample = _sample
    base_tokens = _replace_option(base_tokens, "--distribution-transport-mode", "marginal")
    sys.argv = [sys.argv[0], *base_tokens]
    base.main()

    output = Path(output_value)
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["distribution_transport_mode"] = "none"
    metadata["heldout_used_for_fitting"] = False
    metadata["factor_mmd_fit_source"] = "training_split_only"
    metadata["factor_mmd"] = _OPERATOR.metadata() if _OPERATOR is not None else None
    metadata["mechanism"] = (
        "typed factorized RFF-MMD witness with sparse row-factor middle guidance"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    with (output / "RESULTS.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Factorized RFF-MMD\n\n"
            f"- Mode: `{mmd_args.factor_mmd_mode}`\n"
            f"- Target read-only: `{target_column}`\n"
            f"- Heads retained: `{len(_OPERATOR.heads) if _OPERATOR else 0}`\n"
            "- Reference moments: training split only.\n"
        )


if __name__ == "__main__":
    main()
