#!/usr/bin/env python3
"""Typed state-conditioned graph guidance for pretrained TabDiff.

This experiment isolates the contribution of graph topology.  A train-only
candidate bank is re-verified on disjoint discovery/calibration partitions.
Every online method uses the same typed minimal-projection actuator; only the
edge activation schedule changes.  Test data is used for evaluation only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

from conditioned_tabdiff_adapter import decode, seed_all
from general_state_graph_probe import evaluate, parse_implication, PREDICATE, state_stats
from subgroup_projection_online import PreviewModel
from tabdiff_probe import preprocess_dataset


def split_predicate(text: str) -> tuple[str, str, str]:
    match = PREDICATE.match(str(text))
    if not match:
        raise ValueError(text)
    return tuple(value.strip() for value in match.groups())


def verify(frame: pd.DataFrame, constraint: str) -> dict[str, float]:
    left, right = parse_implication(constraint)
    return state_stats(evaluate(frame, left), evaluate(frame, right))


def discover_graph(catalog: pd.DataFrame, train: pd.DataFrame, dataset: str,
                   maximum: int, seed: int) -> pd.DataFrame:
    """Select without consulting heldout statistics or target/sensitive edges."""
    pool = catalog[
        catalog.dataset.eq(dataset)
        & catalog.source.eq("train_only_predicate_mining")
        & ~catalog.sensitive_attribute.astype(bool)
        & ~catalog.target_related.astype(bool)
    ].copy()
    pool = pool[~pool.constraint.astype(str).str.contains("<MISSING_LIKE>", regex=False)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    cut = int(0.7 * len(train))
    discovery, calibration = train.iloc[order[:cut]], train.iloc[order[cut:]]
    rows = []
    minimum = max(30, int(0.003 * len(discovery)))
    for item in pool.drop_duplicates("constraint").itertuples(index=False):
        try:
            d, c = verify(discovery, item.constraint), verify(calibration, item.constraint)
            left, right = parse_implication(item.constraint)
            a_col, _, _ = split_predicate(left); c_col, _, _ = split_predicate(right)
        except Exception:
            continue
        if min(d["support"], c["support"]) < minimum:
            continue
        if max(d["base_rate"], c["base_rate"]) > .995:
            continue
        worst_v = max(d["violation_rate"], c["violation_rate"])
        min_conf = min(d["confidence"], c["confidence"])
        min_lift = min(d["lift"], c["lift"])
        if worst_v <= .01 and min_lift >= 1.01:
            hardness, weight = "hard", 1.0
        elif worst_v <= .10 and min_conf >= .90 and min_lift >= 1.02:
            hardness, weight = "near_hard", .75
        elif min_conf >= .70 and min_lift >= 1.20:
            hardness, weight = "soft", .40
        else:
            continue
        stability = 1 - abs(d["violation_rate"] - c["violation_rate"])
        coverage = min(d["support"] / len(discovery), c["support"] / len(calibration))
        nonvacuity = max(0.01, 1 - max(d["base_rate"], c["base_rate"]))
        score = math.log1p(min(d["support"], c["support"])) * min(min_lift, 6) * stability * math.sqrt(nonvacuity)
        rows.append({
            "constraint_id": item.constraint_id, "constraint": item.constraint,
            "antecedent_column": a_col, "consequent_column": c_col,
            "hardness": hardness, "actuation_weight": weight, "score": score,
            "discovery_support": d["support"], "discovery_violation": d["violation_rate"],
            "calibration_support": c["support"], "calibration_violation": c["violation_rate"],
            "calibration_confidence": c["confidence"], "calibration_lift": c["lift"],
            "calibration_coverage": coverage,
        })
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("no calibrated graph edges")
    candidates = candidates.sort_values(["hardness", "score"], ascending=[True, False])
    selected, pair_counts = [], {}
    # Preserve endpoint diversity; duplicated threshold variants are not useful graph edges.
    for row in candidates.sort_values("score", ascending=False).itertuples(index=False):
        pair = (row.antecedent_column, row.consequent_column)
        if pair_counts.get(pair, 0) >= 2:
            continue
        selected.append(row._asdict()); pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if len(selected) >= maximum:
            break
    result = pd.DataFrame(selected).sort_values("score", ascending=False).reset_index(drop=True)
    result.insert(0, "edge_index", np.arange(len(result)))
    return result


def load_unified_graph(path: Path) -> pd.DataFrame:
    """Load the source-agnostic graph emitted by unify_verified_specification_graph.py."""
    rows = []
    for index, item in enumerate(json.loads(path.read_text())):
        def render(node: dict) -> str:
            value = node["value"]
            if isinstance(value, float) and value.is_integer(): value = int(value)
            return f"{node['column']} {node['operator']} {value}"
        antecedent, consequent = render(item["antecedent"]), render(item["consequent"])
        rows.append({"edge_index": index, "constraint_id": f"unified:{index}",
                     "constraint": f"IF {antecedent} THEN {consequent}",
                     "antecedent_column": item["antecedent"]["column"],
                     "consequent_column": item["consequent"]["column"],
                     "hardness": item["hardness"],
                     "actuation_weight": {"hard": 1.0, "near_hard": .75, "soft": .4}[item["hardness"]],
                     "score": item["score"], "provenance": "+".join(sorted(item["provenance"])),
                     "discovery_support": item["discovery"]["support"],
                     "discovery_violation": item["discovery"]["violation_rate"],
                     "calibration_support": item["calibration"]["support"],
                     "calibration_violation": item["calibration"]["violation_rate"],
                     "calibration_confidence": item["calibration"]["confidence"],
                     "calibration_lift": item["calibration"]["lift"]})
    if not rows: raise RuntimeError(f"empty unified graph: {path}")
    return pd.DataFrame(rows)


class Actuator:
    """Compile typed consequents into minimal encoded-space projections."""
    def __init__(self, graph: pd.DataFrame, train_data, train: pd.DataFrame, info: dict):
        self.graph = graph
        self.train_data = train_data
        self.numeric_names = [info["column_names"][i] for i in info["num_col_idx"]]
        self.categorical_names = [info["column_names"][i] for i in info["target_col_idx"] + info["cat_col_idx"]]
        self.num_pos = {name: i for i, name in enumerate(self.numeric_names)}
        self.cat_pos = {name: i for i, name in enumerate(self.categorical_names)}
        self.numeric_scale = {
            name: max(float(pd.to_numeric(train[name], errors="coerce").std()), 1e-6)
            for name in self.numeric_names
        }
        self.compiled = []
        for row in graph.itertuples(index=False):
            left, right = parse_implication(row.constraint)
            column, operator, literal = split_predicate(right)
            item = row._asdict() | {"antecedent": left, "consequent": right,
                                    "operator": operator, "literal": literal}
            if column in self.num_pos:
                raw = pd.to_numeric(train[column], errors="coerce").dropna().to_numpy(float)
                threshold = float(literal)
                if operator == ">":
                    possible = raw[raw > threshold]; threshold = float(possible.min()) if len(possible) else np.nextafter(threshold, np.inf)
                elif operator == "<":
                    possible = raw[raw < threshold]; threshold = float(possible.max()) if len(possible) else np.nextafter(threshold, -np.inf)
                probe = train[self.numeric_names].median(numeric_only=True).to_numpy(float)[None, :]
                probe[0, self.num_pos[column]] = threshold
                item.update(target_kind="numeric", target_position=self.num_pos[column],
                            encoded_value=float(train_data.num_transform.transform(probe)[0, self.num_pos[column]]),
                            raw_value=threshold)
            elif column in self.cat_pos and operator == "==":
                categories = train_data.cat_transform.named_steps["ordinal"].categories_[self.cat_pos[column]]
                matches = np.flatnonzero(np.asarray(categories).astype(str) == str(literal))
                if len(matches) != 1:
                    continue
                item.update(target_kind="categorical", target_position=self.cat_pos[column],
                            encoded_value=int(matches[0]), raw_value=str(literal))
            else:
                continue
            self.compiled.append(item)

    def project(self, numeric: torch.Tensor, categorical: torch.Tensor, rule: dict,
                rows: torch.Tensor, strength: float) -> None:
        if not rows.any():
            return
        position, value, operator = rule["target_position"], rule["encoded_value"], rule["operator"]
        if rule["target_kind"] == "categorical":
            categorical[rows, position] = int(value)
            return
        current = numeric[rows, position]
        target = torch.full_like(current, float(value))
        amount = float(strength) * float(rule["actuation_weight"])
        if operator in {">", ">="}:
            projected = torch.maximum(current, target)
        elif operator in {"<", "<="}:
            projected = torch.minimum(current, target)
        else:
            projected = target
        numeric[rows, position] = current + amount * (projected - current)


def phase_allowed(mode: str, t: float) -> bool:
    if mode == "static_all": return True
    if mode in {"fixed_middle", "dynamic_middle", "oracle_middle"}: return .10 <= t <= .50
    if mode in {"adaptive_dynamic", "calibrated_adaptive"}: return .10 <= t <= .75
    if mode == "fixed_late": return t <= .25
    return False


def learn_readability_schedule(path: Path | None, precision: float, recall: float) -> dict[str, float]:
    """Find the earliest reverse-time state at which a violation edge is readable."""
    if path is None: return {}
    trajectory = pd.read_csv(path)
    schedule = {}
    for constraint, block in trajectory.groupby("constraint"):
        eligible = block[
            block.violation_precision.fillna(0).ge(precision)
            & block.violation_recall.fillna(0).ge(recall)
        ]
        # Reverse sampling moves from t=1 to t=0, so the largest qualifying t
        # is the earliest state at which this edge can safely be considered.
        if not eligible.empty: schedule[str(constraint)] = float(eligible.t.max())
    return schedule


@torch.no_grad()
def sample_guided(diffusion, preview: PreviewModel, train_data, info: dict, actuator: Actuator,
                  mode: str, samples: int, strength: float, min_support: int,
                  stable_fraction: float, drift: float,
                  oracle_masks: dict[int, np.ndarray] | None = None,
                  readability_schedule: dict[str, float] | None = None,
                  fallback_cutoff: float = .25) -> tuple[torch.Tensor, dict]:
    device = diffusion.device
    times = torch.linspace(0, 1, diffusion.num_timesteps, dtype=torch.float32, device=device)[:, None]
    sn = diffusion.num_schedule.total_noise(times); sc = diffusion.cat_schedule.total_noise(times)
    snx = torch.zeros_like(sn); snx[1:] = sn[:-1]
    scx = torch.zeros_like(sc); scx[1:] = sc[:-1]
    numeric = torch.randn((samples, diffusion.num_numerical_features), device=device) * sn[-1]
    categorical = diffusion._sample_masked_prior(samples, len(diffusion.num_classes))
    previous_masks: dict[int, np.ndarray] = {}
    previous_numeric = None; previous_codes = None
    stats = {"steps_in_window": 0, "edge_step_enabled": 0, "row_actions": 0,
             "conflicts": 0, "candidate_violations": 0, "edge_trace": []}
    for index in reversed(range(diffusion.num_timesteps)):
        numeric, categorical, _ = diffusion.edm_update(
            numeric, categorical, index, times[index], times[index - 1] if index > 0 else None, times[index],
            sn[index], snx[index], sn[index], sc[index], scx[index], sc[index])
        t = float(times[index])
        if not phase_allowed(mode, t):
            previous_numeric = preview.numeric.detach().cpu().numpy()
            previous_codes = preview.codes.detach().cpu().numpy()
            continue
        stats["steps_in_window"] += 1
        encoded = torch.cat([preview.numeric, preview.codes.float()], dim=1)
        frame = decode(train_data, encoded, info)
        current_numeric = preview.numeric.detach().cpu().numpy()
        current_codes = preview.codes.detach().cpu().numpy()
        claimed = {("numeric", p): torch.zeros(samples, dtype=torch.bool, device=device) for p in actuator.num_pos.values()}
        claimed.update({("categorical", p): torch.zeros(samples, dtype=torch.bool, device=device) for p in actuator.cat_pos.values()})
        trace = {"t": t, "enabled_edges": 0, "actions": 0}
        for edge_index, rule in enumerate(actuator.compiled):
            if mode in {"adaptive_dynamic", "calibrated_adaptive"}:
                cutoff = (readability_schedule or {}).get(rule["constraint"], fallback_cutoff)
                if t > cutoff:
                    continue
            antecedent = evaluate(frame, rule["antecedent"])
            violated = antecedent & ~evaluate(frame, rule["consequent"])
            stats["candidate_violations"] += int(violated.sum())
            enabled = True
            gate = violated.copy()
            if mode in {"dynamic_middle", "adaptive_dynamic", "calibrated_adaptive"}:
                previous = previous_masks.get(edge_index)
                persistent = antecedent & previous if previous is not None else np.zeros(samples, dtype=bool)
                columns_stable = np.ones(samples, dtype=bool)
                if previous_numeric is None:
                    columns_stable[:] = False
                else:
                    for column in {rule["antecedent_column"], rule["consequent_column"]}:
                        if column in actuator.num_pos:
                            pos = actuator.num_pos[column]
                            columns_stable &= np.abs(current_numeric[:, pos] - previous_numeric[:, pos]) <= drift
                        elif column in actuator.cat_pos:
                            pos = actuator.cat_pos[column]
                            columns_stable &= current_codes[:, pos] == previous_codes[:, pos]
                stable = persistent & columns_stable
                support = int(antecedent.sum())
                fraction = float(stable[antecedent].mean()) if support else 0.0
                enabled = support >= min_support and fraction >= stable_fraction
                gate &= stable if enabled else False
            elif mode == "oracle_middle":
                gate &= oracle_masks.get(edge_index, np.zeros(samples, dtype=bool))
            previous_masks[edge_index] = antecedent
            if not enabled or not gate.any():
                continue
            if mode == "calibrated_adaptive" and rule["hardness"] != "hard":
                # Preserve the calibrated exception mass of near-hard edges.
                # Only excess violations receive an action.
                allowed = int(round(float(rule["calibration_violation"]) * int(antecedent.sum())))
                budget = max(0, int(violated.sum()) - allowed)
                indices = np.flatnonzero(gate)
                if budget < len(indices):
                    limited = np.zeros_like(gate); limited[indices[:budget]] = True; gate = limited
                if not gate.any():
                    continue
            key = (rule["target_kind"], rule["target_position"])
            rows = torch.as_tensor(gate, device=device) & ~claimed[key]
            stats["conflicts"] += int((torch.as_tensor(gate, device=device) & claimed[key]).sum().item())
            actuator.project(numeric, categorical, rule, rows, strength)
            claimed[key] |= rows
            acted = int(rows.sum().item())
            stats["row_actions"] += acted; stats["edge_step_enabled"] += 1
            trace["enabled_edges"] += 1; trace["actions"] += acted
        stats["edge_trace"].append(trace)
        previous_numeric = current_numeric; previous_codes = current_codes
    if not torch.all(categorical < diffusion.mask_index):
        raise RuntimeError("categorical mask remained")
    return torch.cat([numeric, categorical], dim=1).cpu(), stats


def repair(frame: pd.DataFrame, actuator: Actuator) -> tuple[pd.DataFrame, float]:
    result = frame.copy(); changed = np.zeros(len(result), dtype=bool)
    # Iterate because one consequent can activate another edge. A bounded
    # fixed-point pass is a stronger and fairer post-repair baseline.
    for _ in range(4):
        edits = 0
        for rule in actuator.compiled:
            gate = evaluate(result, rule["antecedent"]) & ~evaluate(result, rule["consequent"])
            if not gate.any(): continue
            result.loc[gate, rule["consequent_column"]] = rule["raw_value"]
            changed |= gate; edits += int(gate.sum())
        if edits == 0: break
    return result, float(changed.mean())


def graph_metrics(frame: pd.DataFrame, reference: pd.DataFrame, graph: pd.DataFrame) -> dict[str, float]:
    rates, hard_rates, supports, violations, calibration_errors, conditional_distances = [], [], 0, 0, [], []
    for row in graph.itertuples(index=False):
        s = verify(frame, row.constraint)
        r = verify(reference, row.constraint)
        if s["support"]:
            rates.append(s["violation_rate"]); supports += s["support"]; violations += s["violation"]
            if row.hardness in {"hard", "near_hard"}: hard_rates.append(s["violation_rate"])
        if s["support"] and r["support"]:
            calibration_errors.append(abs(s["violation_rate"] - r["violation_rate"]))
            left, _ = parse_implication(row.constraint)
            sample_mask, ref_mask = evaluate(frame, left), evaluate(reference, left)
            column = row.consequent_column
            if pd.api.types.is_numeric_dtype(reference[column]):
                conditional_distances.append(ks_2samp(
                    pd.to_numeric(frame.loc[sample_mask, column]),
                    pd.to_numeric(reference.loc[ref_mask, column])).statistic)
            else:
                a = frame.loc[sample_mask, column].astype(str).value_counts(normalize=True)
                b = reference.loc[ref_mask, column].astype(str).value_counts(normalize=True)
                keys = a.index.union(b.index)
                conditional_distances.append(.5 * float((a.reindex(keys, fill_value=0)-b.reindex(keys, fill_value=0)).abs().sum()))
    return {"macro_violation": float(np.mean(rates)) if rates else math.nan,
            "hard_near_violation": float(np.mean(hard_rates)) if hard_rates else math.nan,
            "micro_violation": violations / supports if supports else math.nan,
            "violation_calibration_error": float(np.mean(calibration_errors)) if calibration_errors else math.nan,
            "conditional_endpoint_distance": float(np.mean(conditional_distances)) if conditional_distances else math.nan,
            "applicable_pairs": supports}


def evaluate_method(frame: pd.DataFrame, base: pd.DataFrame, heldout: pd.DataFrame,
                    graph: pd.DataFrame, info: dict, method: str, stats: dict) -> dict:
    numeric = [info["column_names"][i] for i in info["num_col_idx"]]
    categorical = [info["column_names"][i] for i in info["cat_col_idx"]]
    num_ks = [ks_2samp(pd.to_numeric(frame[c]), pd.to_numeric(heldout[c])).statistic for c in numeric]
    tv = []
    for column in categorical:
        a = frame[column].astype(str).value_counts(normalize=True); b = heldout[column].astype(str).value_counts(normalize=True)
        keys = a.index.union(b.index); tv.append(.5 * float((a.reindex(keys, fill_value=0)-b.reindex(keys, fill_value=0)).abs().sum()))
    rare = []
    for column in categorical:
        counts = heldout[column].astype(str).value_counts()
        required = set(counts[(counts >= 5) & (counts <= max(5, .05 * len(heldout)))].index)
        if required: rare.append(len(required & set(frame[column].astype(str))) / len(required))
    num_drift = np.mean([
        np.mean(np.abs(pd.to_numeric(frame[c]).to_numpy()-pd.to_numeric(base[c]).to_numpy())) /
        max(float(pd.to_numeric(heldout[c]).std()), 1e-6) for c in numeric])
    cat_drift = np.mean([(frame[c].astype(str).to_numpy()!=base[c].astype(str).to_numpy()).mean() for c in categorical]) if categorical else 0
    rounded = frame.copy()
    for c in numeric: rounded[c] = pd.to_numeric(rounded[c]).round(2)
    return {"method": method, **graph_metrics(frame, heldout, graph), "numeric_ks": float(np.mean(num_ks)),
            "categorical_tv": float(np.mean(tv)) if tv else 0.0,
            "rare_category_coverage": float(np.mean(rare)) if rare else math.nan,
            "numeric_drift_sd": float(num_drift), "categorical_change": float(cat_drift),
            "duplicate_rate": float(rounded.duplicated().mean()),
            "row_actions": stats.get("row_actions", 0), "edge_step_enabled": stats.get("edge_step_enabled", 0),
            "conflicts": stats.get("conflicts", 0), "final_repair_rate": stats.get("final_repair_rate", 0.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--dataset", required=True)
    parser.add_argument("--catalog", type=Path, required=True); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unified-graph", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048); parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-edges", type=int, default=12); parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--actuation-level", choices=["hard", "hard_near", "all"], default="hard_near")
    parser.add_argument("--min-support", type=int, default=12); parser.add_argument("--stable-fraction", type=float, default=.70)
    parser.add_argument("--drift", type=float, default=.25); parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--trajectory-calibration", type=Path)
    parser.add_argument("--readability-precision", type=float, default=.90)
    parser.add_argument("--readability-recall", type=float, default=.80)
    parser.add_argument("--fallback-cutoff", type=float, default=.25)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    started = time.perf_counter(); seed_all(args.seed); sys.path.insert(0, str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP
    info = json.loads((args.repo/"data"/args.dataset/"info.json").read_text())
    train_data, test_data = preprocess_dataset(args.repo, args.dataset)
    train, heldout = decode(train_data, train_data.X, info), decode(train_data, test_data.X, info)
    if args.unified_graph:
        graph = load_unified_graph(args.unified_graph)
        graph["selection_priority"] = graph.hardness.map({"hard": 2, "near_hard": 1, "soft": 0})
        graph = graph.sort_values(["selection_priority", "score"], ascending=[False, False]).head(args.max_edges).copy()
        graph = graph.drop(columns="selection_priority").reset_index(drop=True)
        graph["edge_index"] = np.arange(len(graph))
    else:
        graph = discover_graph(pd.read_csv(args.catalog), train, args.dataset, args.max_edges, args.seed)
    allowed = {"hard"} if args.actuation_level == "hard" else ({"hard", "near_hard"} if args.actuation_level == "hard_near" else {"hard", "near_hard", "soft"})
    graph["actuated"] = graph.hardness.isin(allowed)
    actuator = Actuator(graph[graph.actuated].copy(), train_data, train, info)
    graph["actuatable"] = graph.edge_index.isin([r["edge_index"] for r in actuator.compiled])
    graph.to_csv(args.output/"verified_specification_graph.csv", index=False)
    if not actuator.compiled: raise RuntimeError("no actuatable graph edges")
    with args.config.open("rb") as handle: config = pickle.load(handle)
    params = copy.deepcopy(config["unimodmlp_params"]); params["d_numerical"] = train_data.d_numerical
    params["categories"] = (train_data.categories + 1).tolist()
    model = Model(UniModMLP(**params), **config["diffusion_params"]["edm_params"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False); model.load_state_dict(state["denoise_fn"], strict=True)
    preview = PreviewModel(model, train_data.categories)
    dp = copy.deepcopy(config["diffusion_params"]); dp.update(num_timesteps=args.steps, scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column")
    dp["sampler_params"] = copy.deepcopy(dp["sampler_params"]); dp["sampler_params"].update(stochastic_sampler=False, second_order_correction=False)
    diffusion = UnifiedCtimeDiffusion(train_data.categories, train_data.d_numerical, preview, None, **dp, device=torch.device("cuda")).cuda()
    diffusion.num_schedule.load_state_dict(state["num_schedule"]); diffusion.cat_schedule.load_state_dict(state["cat_schedule"]); diffusion.eval()
    common_seed = args.seed + 100000
    seed_all(common_seed); base = decode(train_data, diffusion.sample(args.samples), info)
    oracle_masks = {}
    for edge_index, rule in enumerate(actuator.compiled):
        oracle_masks[edge_index] = evaluate(base, rule["antecedent"]) & ~evaluate(base, rule["consequent"])
    rows = [evaluate_method(base, base, heldout, graph, info, "unconditional", {})]
    outputs = {}
    readability_schedule = learn_readability_schedule(
        args.trajectory_calibration, args.readability_precision, args.readability_recall)
    pd.DataFrame([{"constraint": key, "cutoff": value} for key, value in readability_schedule.items()]).to_csv(
        args.output/"learned_readability_schedule.csv", index=False)
    for mode in ["static_all", "fixed_middle", "fixed_late", "dynamic_middle", "adaptive_dynamic", "calibrated_adaptive", "oracle_middle"]:
        seed_all(common_seed)
        encoded, diagnostics = sample_guided(diffusion, preview, train_data, info, actuator, mode,
            args.samples, args.strength, args.min_support, args.stable_fraction, args.drift, oracle_masks,
            readability_schedule, args.fallback_cutoff)
        frame = decode(train_data, encoded, info); outputs[mode] = frame
        rows.append(evaluate_method(frame, base, heldout, graph, info, mode, diagnostics))
        (args.output/f"trace_{mode}.json").write_text(json.dumps(diagnostics, indent=2))
    repaired, rate = repair(base, actuator)
    rows.append(evaluate_method(repaired, base, heldout, graph, info, "final_repair", {"final_repair_rate": rate}))
    certified, rate = repair(outputs["dynamic_middle"], actuator)
    rows.append(evaluate_method(certified, base, heldout, graph, info, "dynamic_middle_plus_certifier", {"final_repair_rate": rate}))
    certified, rate = repair(outputs["adaptive_dynamic"], actuator)
    rows.append(evaluate_method(certified, base, heldout, graph, info, "adaptive_dynamic_plus_certifier", {"final_repair_rate": rate}))
    metrics = pd.DataFrame(rows); metrics.insert(0, "dataset", args.dataset); metrics.to_csv(args.output/"method_metrics.csv", index=False)
    (args.output/"run_metadata.json").write_text(json.dumps({
        "dataset": args.dataset, "samples": args.samples, "steps": args.steps, "seed": args.seed,
        "selected_edges": len(graph), "actuatable_edges": len(actuator.compiled), "actuation_level": args.actuation_level,
        "strength": args.strength,
        "min_support": args.min_support, "stable_fraction": args.stable_fraction, "drift": args.drift,
        "unified_graph": str(args.unified_graph) if args.unified_graph else None,
        "trajectory_calibration": str(args.trajectory_calibration) if args.trajectory_calibration else None,
        "readability_schedule_edges": len(readability_schedule),
        "readability_precision": args.readability_precision, "readability_recall": args.readability_recall,
        "selection_uses_heldout": False, "same_noise_graph_and_actuator": True,
        "runtime_seconds": time.perf_counter()-started}, indent=2))
    print(graph[["constraint", "hardness", "calibration_violation"]].to_string(index=False))
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
