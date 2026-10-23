#!/usr/bin/env python3
"""Train-only temporal edge routing with late conditional diffusion actions.

Each edge is routed to the middle state with the largest positive precursor
surplus (middle-only minus final-only violations) on an unlabeled calibration
trajectory.  Otherwise it is deferred to the final state.  Activation and
action generation are intentionally decoupled: all correction proposals are
conditioned on the cleaner final state.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conditioned_tabdiff_adapter import decode
from dynamic_graph_conditional_repaint import FrozenProposalScore, eligible_edges, requirement_target
from graph_conditioned_residual_denoiser import calibrated_minimal_final_repair, load_backbone, seed_all
from trajectory_graph_conditional_repaint import sample_base_and_previews
from typed_dynamic_graph_guidance import TypedActuator, edge_masks, graph_metrics


def learn_edge_routes(edges: list[dict], calibration_dir: Path, checkpoints: list[float]) -> tuple[dict, dict]:
    final = pd.read_csv(calibration_dir / "tabdiff.csv")
    middle = {value: pd.read_csv(calibration_dir / f"trigger_preview_{value:.2f}.csv") for value in checkpoints}
    routes = {}; diagnostics = {}
    for edge in edges:
        edge_id = edge["candidate_id"]; final_bad = edge_masks(final, edge)[2]
        candidates = []
        for value in checkpoints:
            middle_bad = edge_masks(middle[value], edge)[2]
            middle_only = int((middle_bad & ~final_bad).sum())
            final_only = int((~middle_bad & final_bad).sum())
            candidates.append({"state": value, "middle_only": middle_only, "final_only": final_only,
                               "surplus": middle_only - final_only, "persistent": int((middle_bad & final_bad).sum())})
        best = max(candidates, key=lambda item: (item["surplus"], item["state"]))
        routes[edge_id] = best["state"] if best["surplus"] > 0 else "final"
        diagnostics[edge_id] = {"route": routes[edge_id], "candidates": candidates,
                                "final_violations": int(final_bad.sum())}
    return routes, diagnostics


def routed_target_groups(frames: dict, edges: list[dict], routes: dict,
                         numeric_names: list[str], categorical_names: list[str], max_targets: int):
    priorities = {"hard": 3, "near_hard": 2, "soft": 1}; per_row = defaultdict(list)
    triggered = Counter()
    for edge in edges:
        target = requirement_target(edge, numeric_names, categorical_names)
        if target is None: continue
        route = routes[edge["candidate_id"]]; frame = frames[route]
        bad = edge_masks(frame, edge)[2]; triggered[f"{edge['candidate_id']}@{route}"] = int(bad.sum())
        priority = priorities.get(edge.get("classification", "soft"), 1)
        confidence = float(edge.get("confidence", 0.0))
        for row in np.flatnonzero(bad): per_row[int(row)].append((priority, confidence, target))
    groups = defaultdict(list)
    for row, candidates in per_row.items():
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2])); targets=[]
        for _, _, target in candidates:
            if target not in targets: targets.append(target)
            if len(targets) >= max_targets: break
        if targets: groups[tuple(sorted(targets))].append(row)
    return {key: np.asarray(value, dtype=int) for key, value in groups.items()}, dict(triggered)


@torch.no_grad()
def repaint(diffusion, train_data, info, base_frame, final_encoded, frames, edges, routes,
            scorer, branches, rounds, max_targets, seed, accept_margin):
    groups, triggered = routed_target_groups(
        frames, edges, routes, scorer.numeric_names, scorer.categorical_names, max_targets)
    result=base_frame.copy(); choices=Counter(); generated=0; started=time.perf_counter(); group_rows={}
    for ordinal, (targets, indices) in enumerate(sorted(groups.items())):
        subset=final_encoded[indices]
        num_positions=[scorer.numeric_names.index(name) for name in targets if name in scorer.numeric_names]
        cat_positions=[scorer.categorical_names.index(name) for name in targets if name in scorer.categorical_names]
        base_subset=base_frame.iloc[indices].reset_index(drop=True); proposals=[base_subset.copy()]
        for branch in range(branches):
            seed_all(seed + ordinal * 1009 + branch)
            sampled=diffusion.sample_impute(
                subset[:, :train_data.d_numerical].to(diffusion.device),
                subset[:, train_data.d_numerical:].long().to(diffusion.device),
                num_positions, cat_positions, rounds, "x_t", 0.0, 0.0)
            decoded=decode(train_data, sampled, info); candidate=base_subset.copy()
            for name in targets: candidate[name]=decoded[name].to_numpy()
            proposals.append(candidate)
        scores=np.stack([scorer(base_subset, proposal, targets) for proposal in proposals], 0)
        scores[~np.isfinite(scores)]=np.inf; choice=np.argmin(scores, axis=0)
        improvement=scores[0]-scores[choice, np.arange(len(indices))]; choice[improvement < accept_margin]=0
        for proposal_index, proposal in enumerate(proposals):
            selected=choice == proposal_index
            if not selected.any(): continue
            choices[str(proposal_index)] += int(selected.sum())
            if proposal_index:
                destination=indices[selected]
                for name in targets: result.loc[destination, name]=proposal.loc[selected, name].to_numpy()
        generated += branches * len(indices); group_rows["|".join(targets)] = int(len(indices))
    return result, {"triggered_edges":triggered, "target_groups":len(groups), "group_rows":group_rows,
                    "proposal_rows_generated":generated, "branch_choices":dict(choices),
                    "unique_rows_changed":int(sum(v for k,v in choices.items() if k != "0")),
                    "runtime_seconds":time.perf_counter()-started}


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True); p.add_argument("--dataset",required=True)
    p.add_argument("--config",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--graph",type=Path,required=True); p.add_argument("--calibration-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--checkpoints",default=".40,.25")
    p.add_argument("--samples",type=int,default=4096); p.add_argument("--steps",type=int,default=50)
    p.add_argument("--branches",type=int,default=8); p.add_argument("--rounds",type=int,default=2)
    p.add_argument("--max-targets",type=int,default=3); p.add_argument("--max-calibration-rate",type=float,default=.05)
    p.add_argument("--graph-weight",type=float,default=20.0); p.add_argument("--density-weight",type=float,default=1.0)
    p.add_argument("--rarity-weight",type=float,default=.1); p.add_argument("--edit-weight",type=float,default=.5)
    p.add_argument("--accept-margin",type=float,default=0.0); p.add_argument("--density-bank",type=int,default=20000)
    p.add_argument("--seed",type=int,default=20261444); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True); sample_dir=a.output/"samples"; sample_dir.mkdir()
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    checkpoints=[float(value) for value in a.checkpoints.split(",")]; started=time.perf_counter()
    diffusion,train_data,test_data,info,_=load_backbone(a.repo,a.dataset,a.config,a.checkpoint,torch.device("cuda"),a.steps)
    all_edges=json.loads(a.graph.read_text()); edges=eligible_edges(all_edges,a.max_calibration_rate)
    routes,route_diagnostics=learn_edge_routes(edges,a.calibration_dir,checkpoints)
    train=decode(train_data,train_data.X,info); heldout=decode(train_data,test_data.X,info)
    base_encoded,previews,actual,base_runtime=sample_base_and_previews(
        diffusion,a.samples,checkpoints,a.seed+100000)
    base_frame=decode(train_data,base_encoded,info); frames={"final":base_frame}
    for value in checkpoints: frames[value]=decode(train_data,previews[value],info)
    base_frame.to_csv(sample_dir/"tabdiff.csv",index=False)
    for value in checkpoints: frames[value].to_csv(sample_dir/f"trigger_preview_{value:.2f}.csv",index=False)
    scorer=FrozenProposalScore(train_data,train,info,edges,a.graph_weight,a.density_weight,a.rarity_weight,a.edit_weight,a.density_bank)
    routed,diagnostics=repaint(diffusion,train_data,info,base_frame,base_encoded,frames,edges,routes,scorer,
                               a.branches,a.rounds,a.max_targets,a.seed+200000,a.accept_margin)
    routed.to_csv(sample_dir/"calibrated_temporal_router.csv",index=False)
    actuator=TypedActuator(all_edges,train_data,train,info)
    final_repair,final_diag=calibrated_minimal_final_repair(base_frame,actuator)
    routed_repair,routed_diag=calibrated_minimal_final_repair(routed,actuator)
    final_repair.to_csv(sample_dir/"calibrated_final_repair.csv",index=False)
    routed_repair.to_csv(sample_dir/"calibrated_temporal_router_plus_repair.csv",index=False)
    rows=[
      {"method":"tabdiff",**graph_metrics(base_frame,heldout,all_edges),"runtime_seconds":base_runtime},
      {"method":"calibrated_final_repair",**graph_metrics(final_repair,heldout,all_edges),"runtime_seconds":base_runtime,**final_diag},
      {"method":"calibrated_temporal_router",**graph_metrics(routed,heldout,all_edges),"runtime_seconds":base_runtime+diagnostics["runtime_seconds"],**diagnostics},
      {"method":"calibrated_temporal_router_plus_repair",**graph_metrics(routed_repair,heldout,all_edges),"runtime_seconds":base_runtime+diagnostics["runtime_seconds"],**routed_diag},
    ]
    pd.DataFrame(rows).to_csv(a.output/"mechanism_metrics.csv",index=False)
    metadata={"method":"calibrated_temporal_edge_router","dataset":a.dataset,"routes":routes,
              "route_calibration":route_diagnostics,"requested_checkpoints":checkpoints,"actual_checkpoints":actual,
              "selection_uses_heldout":False,"api_calls":0,"diagnostics":diagnostics,
              "runtime_seconds":time.perf_counter()-started,
              "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (a.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2)); print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": main()
