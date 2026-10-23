#!/usr/bin/env python3
"""Test distribution-aware uses of verified external knowledge.

This is a mechanism proxy, not a reproduction of TabKG, LLM-TabLogic, LCATS,
or SAGE.  It deliberately starts from the same unconditional TabDiff sample
and asks whether three ideas from those papers transfer to our datasets:

* reconstruct only dependents of exact, train-verified relations;
* rank repair alternatives by train density and edit cost;
* use value-aware sparse context when reconstructing a dependent value.

The held-out split is used only by the final evaluator.  Edge selection,
nearest-neighbour reconstruction, and sparse-context statistics use train data.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from evaluate_shoppers_baseline_suite import (
    alpha_beta,
    constraint_metrics,
    dcr_score,
    mle_auc,
    quality_and_detection,
)
from rowwise_reverse_action_guidance import RowActionLayer, final_action
from tabdiff_probe import preprocess_dataset
from typed_dynamic_graph_guidance import edge_masks


def train_rate(edge: dict, train: pd.DataFrame) -> tuple[int, float]:
    active, _, violated = edge_masks(train, edge)
    support = int(active.sum())
    return support, float(violated.sum() / support) if support else math.nan


def exact_edges(graph: list[dict], train: pd.DataFrame, include_score_only: bool,
                min_support: int) -> list[dict]:
    selected = []
    allowed_roles = {"actuate", "score_only"} if include_score_only else {"actuate"}
    for original in graph:
        if original.get("usage_role", "actuate") not in allowed_roles:
            continue
        support, rate = train_rate(original, train)
        if support < min_support or not np.isfinite(rate) or rate > 0:
            continue
        edge = copy.deepcopy(original)
        # This proxy explicitly authorizes only the selected exact relations.
        edge["usage_role"] = "actuate"
        edge["proxy_train_support"] = support
        edge["proxy_train_violation_rate"] = rate
        selected.append(edge)
    return selected


class ValueStates:
    """Train-fitted discretization and value-specific dependency scores."""

    def __init__(self, train: pd.DataFrame, max_bins: int = 16):
        self.train = train.reset_index(drop=True)
        self.numeric = {
            column for column in train.columns if pd.api.types.is_numeric_dtype(train[column])
        }
        self.boundaries: dict[str, np.ndarray] = {}
        self.train_states: dict[str, np.ndarray] = {}
        for column in train.columns:
            if column not in self.numeric:
                states = train[column].astype(str).fillna("<NA>").to_numpy()
            else:
                values = pd.to_numeric(train[column], errors="coerce").to_numpy(float)
                finite = values[np.isfinite(values)]
                if len(finite) < 2 or np.nanmax(finite) <= np.nanmin(finite):
                    boundaries = np.asarray([], dtype=float)
                else:
                    q75, q25 = np.percentile(finite, [75, 25])
                    width = 2.0 * (q75 - q25) * (len(finite) ** (-1.0 / 3.0))
                    bins = int(np.ceil((finite.max() - finite.min()) / width)) if width > 0 else 1
                    bins = min(max(bins, 1), max_bins)
                    boundaries = np.linspace(finite.min(), finite.max(), bins + 1)[1:-1]
                self.boundaries[column] = boundaries
                states = self.state(column, values)
            self.train_states[column] = np.asarray(states)

    def state(self, column: str, values) -> np.ndarray:
        if column not in self.numeric:
            return pd.Series(values).astype(str).fillna("<NA>").to_numpy()
        array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
        state = np.digitize(np.nan_to_num(array, nan=-np.inf), self.boundaries[column]).astype(str)
        state[~np.isfinite(array)] = "<NA>"
        return state

    def row_state(self, row: pd.Series, column: str) -> str:
        return str(self.state(column, [row[column]])[0])

    def target_state(self, columns: list[str]) -> np.ndarray:
        values = [self.train_states[column].astype(str) for column in columns]
        return np.asarray(["|".join(items) for items in zip(*values)])

    def value_relevance(self, context: str, state: str, target: np.ndarray) -> float:
        indicator = (self.train_states[context].astype(str) == str(state)).astype(np.int8)
        if indicator.min() == indicator.max():
            return 0.0
        best = 0.0
        for target_state in np.unique(target):
            target_indicator = (target == target_state).astype(np.int8)
            best = max(best, float(mutual_info_score(indicator, target_indicator)))
        return best


def requirement_columns(edge: dict) -> list[str]:
    columns = []
    for atom in edge.get("requirements", []):
        columns.append(atom["left_column"])
    return list(dict.fromkeys(columns))


def nearest_valid_row(row: pd.Series, train: pd.DataFrame, valid_indices: np.ndarray,
                      contexts: list[str], numeric: set[str], scales: dict[str, float],
                      top_k: int, rng: np.random.Generator) -> int:
    distances = np.zeros(len(valid_indices), dtype=float)
    candidates = train.iloc[valid_indices]
    for column in contexts:
        if column in numeric:
            values = pd.to_numeric(candidates[column], errors="coerce").fillna(0).to_numpy(float)
            target = float(pd.to_numeric(pd.Series([row[column]]), errors="coerce").fillna(0).iloc[0])
            distances += ((values - target) / scales[column]) ** 2
        else:
            distances += (candidates[column].astype(str).to_numpy() != str(row[column])).astype(float)
    width = min(max(1, top_k), len(valid_indices))
    nearest = np.argpartition(distances, width - 1)[:width]
    return int(valid_indices[int(rng.choice(nearest))])


def reconstruct(base: pd.DataFrame, train: pd.DataFrame, edges: list[dict], mode: str,
                top_k: int, sparse_context: int, passes: int, seed: int
                ) -> tuple[pd.DataFrame, dict]:
    result = base.copy()
    states = ValueStates(train)
    numeric = states.numeric
    scales = {
        column: max(float(pd.to_numeric(train[column], errors="coerce").std()), 1e-6)
        for column in numeric
    }
    rng = np.random.default_rng(seed)
    cache = {}
    events = 0
    touched = np.zeros(len(result), dtype=bool)
    for pass_index in range(passes):
        changed_this_pass = 0
        for edge in edges:
            target_columns = requirement_columns(edge)
            if not target_columns:
                continue
            active_train, satisfied_train, _ = edge_masks(train, edge)
            valid_indices = np.flatnonzero(active_train & satisfied_train)
            if not len(valid_indices):
                continue
            _, _, violated = edge_masks(result, edge)
            indices = np.flatnonzero(violated)
            if not len(indices):
                continue
            context_pool = [column for column in result.columns if column not in target_columns]
            target_state = states.target_state(target_columns)
            for row_index in indices:
                row = result.iloc[row_index]
                if mode == "sparse_value_mi":
                    scored = []
                    for column in context_pool:
                        key = (edge.get("candidate_id", "edge"), column, states.row_state(row, column))
                        if key not in cache:
                            cache[key] = states.value_relevance(column, key[-1], target_state)
                        scored.append((cache[key], column))
                    contexts = [column for _, column in sorted(scored, reverse=True)[:sparse_context]]
                else:
                    contexts = context_pool
                chosen = nearest_valid_row(
                    row, train, valid_indices, contexts, numeric, scales, top_k, rng
                )
                before = result.loc[result.index[row_index], target_columns].astype(str).tolist()
                result.loc[result.index[row_index], target_columns] = train.iloc[chosen][target_columns].to_numpy()
                after = result.loc[result.index[row_index], target_columns].astype(str).tolist()
                if before != after:
                    events += 1
                    changed_this_pass += 1
                    touched[row_index] = True
        if changed_this_pass == 0:
            break
    return result, {
        "action_events": events,
        "unique_rows_actuated": int(touched.sum()),
        "passes": pass_index + 1,
        "mode": mode,
    }


def evaluate(name: str, sample: pd.DataFrame, base: pd.DataFrame, train: pd.DataFrame,
             heldout: pd.DataFrame, graph: list[dict], info: dict, seed: int, diagnostics: dict) -> dict:
    sample = sample[train.columns].copy()
    shape, trend, detection = quality_and_detection(train, sample, info)
    alpha, beta = alpha_beta(train, sample, info)
    changed = (sample.astype(str) != base.astype(str))
    return {
        "method": name,
        "rows": len(sample),
        "shape_error_percent": shape,
        "trend_error_percent": trend,
        "alpha_precision_percent": alpha,
        "beta_recall_percent": beta,
        "detection_score": detection,
        "mle_xgboost_auc": mle_auc(sample, heldout, info, seed),
        "dcr_percent_50_ideal": dcr_score(train, heldout, sample, info),
        "row_edit_rate": float(changed.any(axis=1).mean()),
        "cell_edit_rate": float(changed.to_numpy().mean()),
        **constraint_metrics(sample, heldout, graph),
        **diagnostics,
    }


def markdown(frame: pd.DataFrame, selected_counts: dict) -> str:
    columns = [
        "method", "shape_error_percent", "trend_error_percent",
        "alpha_precision_percent", "beta_recall_percent", "detection_score",
        "mle_xgboost_auc", "dcr_percent_50_ideal", "row_edit_rate",
        "heldout_violation_error", "action_events",
    ]
    columns = [column for column in columns if column in frame.columns]
    lines = [
        "# External-knowledge mechanism transfer benchmark", "",
        "This is a proxy benchmark on the same unconditional sample, not a faithful",
        "reproduction of any cited paper. All selection/reconstruction uses train only;",
        "the held-out split is evaluator-only.", "",
        f"Selected exact edges: `{json.dumps(selected_counts, sort_keys=True)}`", "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sparse-context", type=int, default=4)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sample_dir = args.output / "samples"
    sample_dir.mkdir()

    info = json.loads((args.repo / "data" / args.dataset / "info.json").read_text())
    train = pd.read_csv(args.repo / "synthetic" / args.dataset / "real.csv")
    heldout = pd.read_csv(args.repo / "synthetic" / args.dataset / "test.csv")
    base = pd.read_csv(args.sample)[train.columns]
    graph = json.loads(args.graph.read_text())
    train_data, _ = preprocess_dataset(args.repo, args.dataset)

    policies = {
        "actuate_exact": exact_edges(graph, train, False, args.min_support),
        "all_exact": exact_edges(graph, train, True, args.min_support),
    }
    selected_counts = {name: len(edges) for name, edges in policies.items()}
    (args.output / "selected_edges.json").write_text(json.dumps(policies, indent=2))

    methods: list[tuple[str, pd.DataFrame, dict]] = [
        ("unconditional", base.copy(), {"action_events": 0, "unique_rows_actuated": 0}),
    ]
    for policy_name, edges in policies.items():
        if not edges:
            methods.extend([
                (f"{policy_name}:lcats_weighted_repair", base.copy(),
                 {"action_events": 0, "unique_rows_actuated": 0, "abstained_no_exact_edges": 1}),
                (f"{policy_name}:tabkg_knn_reconstruct", base.copy(),
                 {"action_events": 0, "unique_rows_actuated": 0, "abstained_no_exact_edges": 1}),
                (f"{policy_name}:sage_sparse_reconstruct", base.copy(),
                 {"action_events": 0, "unique_rows_actuated": 0, "abstained_no_exact_edges": 1}),
            ])
            continue
        layer = RowActionLayer(
            edges, train_data, train, info,
            density_weight=1.0, edit_weight=0.5, graph_weight=1.0,
            action_planner="edge_local", lock_applicability_columns=True,
        )
        repaired, repair_stats = final_action(base, edges, layer, "reward")
        methods.append((f"{policy_name}:lcats_weighted_repair", repaired, repair_stats))
        reconstructed, reconstruct_stats = reconstruct(
            base, train, edges, "all_context", args.top_k, args.sparse_context,
            args.passes, args.seed + 11,
        )
        methods.append((f"{policy_name}:tabkg_knn_reconstruct", reconstructed, reconstruct_stats))
        sparse, sparse_stats = reconstruct(
            base, train, edges, "sparse_value_mi", args.top_k, args.sparse_context,
            args.passes, args.seed + 23,
        )
        methods.append((f"{policy_name}:sage_sparse_reconstruct", sparse, sparse_stats))

    rows = []
    for name, sample, diagnostics in methods:
        safe = name.replace(":", "__")
        sample.to_csv(sample_dir / f"{safe}.csv", index=False)
        rows.append(evaluate(name, sample, base, train, heldout, graph, info, args.seed, diagnostics))
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "metrics.csv", index=False)
    report = markdown(result, selected_counts)
    (args.output / "RESULTS.md").write_text(report)
    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "source_sample": str(args.sample),
        "source_graph": str(args.graph),
        "selected_counts": selected_counts,
        "scope": "mechanism proxies, not paper reproductions",
        "heldout_policy": "evaluation only",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(report)


if __name__ == "__main__":
    main()
