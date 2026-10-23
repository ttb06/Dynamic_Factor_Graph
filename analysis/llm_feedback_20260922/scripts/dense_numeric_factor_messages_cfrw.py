#!/usr/bin/env python3
"""CFRW trajectory driver layered on ``dense_numeric_factor_messages_arr``.

This wrapper deliberately reuses the verified sampler, schedules, logging and
encoded trust region from the ARR driver.  Only the batch-transport operator
is replaced, and only when ``--residual-witness-mode`` is active.  The ordinary
``--distribution-transport-strength/min-t/max-t/schedule/once/max-calls``
arguments control CFRW pulses; all other ARR options remain available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import dense_numeric_factor_messages_arr as base
from factorized_residual_witness import FactorizedResidualWitnessTransport, VALID_MODES


_WITNESS: FactorizedResidualWitnessTransport | None = None


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


def _witness_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Additional CFRW options (ordinary ARR options are unchanged).",
    )
    parser.add_argument(
        "--residual-witness-mode",
        choices=sorted(VALID_MODES),
        default="none",
        help="Factor families to fit; target is always read-only.",
    )
    parser.add_argument("--residual-witness-trees", type=int, default=32)
    parser.add_argument("--residual-witness-min-r2", type=float, default=0.02)
    parser.add_argument("--residual-witness-max-pair-heads", type=int, default=16)
    parser.add_argument("--residual-witness-top-factors-per-row", type=int, default=2)
    parser.add_argument(
        "--residual-witness-top-rows-per-factor",
        type=float,
        default=0.15,
        help="Fraction in (0,1], absolute count >1, or 0 for all rows.",
    )
    parser.add_argument("--residual-witness-min-score", type=float, default=0.25)
    parser.add_argument("--residual-witness-clip", type=float, default=2.0)
    parser.add_argument(
        "--residual-witness-seed",
        type=int,
        default=None,
        help="Head/cross-fit seed; defaults to the run --seed.",
    )
    return parser


def _aggregate_witness_diagnostics(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    trace = diagnostics.get("trace", [])
    witness_steps = [
        step for step in trace
        if any(key.startswith("residual_witness_") for key in step)
    ]
    diagnostics["residual_witness_calls"] = len(witness_steps)
    if not witness_steps:
        return diagnostics
    keys = sorted({
        key for step in witness_steps for key in step
        if key.startswith("residual_witness_")
    })
    additive = {
        "residual_witness_candidate_row_factors",
        "residual_witness_factor_budget_row_factors",
        "residual_witness_row_budget_row_factors",
        "residual_witness_selected_row_factors",
        "residual_witness_selected_rows",
        "residual_witness_selected_cells",
        "residual_witness_conflicting_cells",
        "residual_witness_marginal_selected_row_factors",
        "residual_witness_target_selected_row_factors",
        "residual_witness_pair_selected_row_factors",
        "residual_witness_marginal_proposed_absolute_move",
        "residual_witness_target_proposed_absolute_move",
        "residual_witness_pair_proposed_absolute_move",
    }
    for key in keys:
        raw_values = [step[key] for step in witness_steps if key in step]
        if not raw_values:
            continue
        if all(isinstance(value, (int, float, bool, np.number)) for value in raw_values):
            values = [float(value) for value in raw_values]
            diagnostics[key] = float(
                np.sum(values) if key in additive else np.mean(values)
            )
        else:
            # Head identifiers/reliabilities are static run metadata, but
            # retaining them in trace.json and the mechanism CSV makes every
            # selected factor auditable without opening a pickle.
            diagnostics[key] = raw_values[-1]
    for manifest_key in (
        "witness_retained_head_ids",
        "witness_retained_head_reliabilities",
    ):
        manifests = [
            step[manifest_key] for step in witness_steps if manifest_key in step
        ]
        if manifests:
            diagnostics[manifest_key] = manifests[-1]
    return diagnostics


def main() -> None:
    global _WITNESS
    witness_parser = _witness_parser()
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(witness_parser.format_help())
        print("Base ARR options follow:\n")
    witness_args, base_tokens = witness_parser.parse_known_args(sys.argv[1:])
    if witness_args.residual_witness_mode == "none":
        sys.argv = [sys.argv[0], *base_tokens]
        base.main()
        return

    distribution_mode = _option_value(
        base_tokens, "--distribution-transport-mode", "none"
    )
    if distribution_mode != "none":
        witness_parser.error(
            "active CFRW requires --distribution-transport-mode none; "
            "the two trajectory transports cannot be combined in this driver"
        )
    incompatible = [
        token for token in base_tokens
        if token.startswith("--distribution-transport-condition-target")
        or token.startswith("--distribution-transport-soft-quantiles")
        or token.startswith("--distribution-transport-adaptive-resolution")
    ]
    if incompatible:
        witness_parser.error(
            "CFRW does not accept conditional/adaptive quantile transport flags: "
            + ", ".join(incompatible)
        )
    strength = float(_option_value(
        base_tokens, "--distribution-transport-strength", "0"
    ))
    if strength <= 0:
        witness_parser.error(
            "active CFRW requires positive --distribution-transport-strength"
        )

    repo_value = _option_value(base_tokens, "--repo")
    dataset = _option_value(base_tokens, "--dataset", "news")
    output_value = _option_value(base_tokens, "--output")
    run_seed = int(_option_value(base_tokens, "--seed", "20261601"))
    if repo_value is None or output_value is None:
        witness_parser.error("--repo and --output remain required base options")
    info_path = Path(repo_value) / "data" / str(dataset) / "info.json"
    info = json.loads(info_path.read_text())
    target_indices = list(info.get("target_col_idx", []))
    target_column = (
        info["column_names"][target_indices[0]] if target_indices else None
    )
    witness_seed = (
        run_seed
        if witness_args.residual_witness_seed is None
        else witness_args.residual_witness_seed
    )

    class _WitnessFactory:
        def __new__(
            cls,
            train,
            numeric_columns,
            edges,
            **kwargs,
        ):
            del cls, kwargs
            global _WITNESS
            _WITNESS = FactorizedResidualWitnessTransport(
                train,
                numeric_columns,
                edges,
                mode=witness_args.residual_witness_mode,
                target_column=target_column,
                trees=witness_args.residual_witness_trees,
                min_r2=witness_args.residual_witness_min_r2,
                max_pair_heads=witness_args.residual_witness_max_pair_heads,
                top_factors_per_row=witness_args.residual_witness_top_factors_per_row,
                top_rows_per_factor=witness_args.residual_witness_top_rows_per_factor,
                min_score=witness_args.residual_witness_min_score,
                clip=witness_args.residual_witness_clip,
                seed=witness_seed,
            )
            return _WITNESS

    original_sample = base.sample

    def _sample_with_witness(*args, **kwargs):
        encoded, diagnostics = original_sample(*args, **kwargs)
        return encoded, _aggregate_witness_diagnostics(diagnostics)

    # Force the existing non-conditional construction branch, while retaining
    # the user's original "none" choice in postprocessed metadata.  No released
    # sampler, TabDiff model, main.py or trainer.py is modified.
    base.BatchDistributionTransport = _WitnessFactory
    base.sample = _sample_with_witness
    base_tokens = _replace_option(
        base_tokens, "--distribution-transport-mode", "marginal"
    )
    sys.argv = [sys.argv[0], *base_tokens]
    base.main()

    output = Path(output_value)
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["distribution_transport_mode"] = "none"
    metadata["heldout_used_for_fitting"] = False
    metadata["residual_witness_fit_source"] = "training_split_only"
    metadata["residual_witness"] = (
        _WITNESS.metadata() if _WITNESS is not None else None
    )
    metadata["residual_witness_crossfit_folds"] = sorted({
        int(head.crossfit_folds)
        for head in (_WITNESS.heads if _WITNESS is not None else [])
        if head.crossfit_folds > 0
    })
    metadata["mechanism"] = (
        "cross-fitted factorized residual witness with sparse row-factor routing"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    results_path = output / "RESULTS.md"
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## CFRW\n\n"
            f"- Mode: `{witness_args.residual_witness_mode}`\n"
            f"- Target read-only: `{target_column}`\n"
            f"- Heads retained: `{len(_WITNESS.heads) if _WITNESS else 0}`\n"
            "- Fitting source: training split only; conditional reliability: OOF.\n"
        )


if __name__ == "__main__":
    main()
