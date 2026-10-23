#!/usr/bin/env python3
"""State-dependent typed-edge activation during pretrained TabDiff sampling."""
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
from knowledge_graph_protocol import predicate_mask
from subgroup_projection_online import PreviewModel
from tabdiff_probe import preprocess_dataset


def edge_masks(frame: pd.DataFrame, edge: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.ones(len(frame), dtype=bool)
    for atom in edge.get("applicability", []): active &= predicate_mask(frame, atom)
    exception = np.zeros(len(frame), dtype=bool)
    for group in edge.get("exceptions", []):
        group_mask = np.ones(len(frame), dtype=bool)
        for atom in group.get("all_of", []): group_mask &= predicate_mask(frame, atom)
        exception |= group_mask
    active &= ~exception
    satisfied = np.ones(len(frame), dtype=bool)
    for atom in edge.get("requirements", []): satisfied &= predicate_mask(frame, atom)
    return active, satisfied, active & ~satisfied


class TypedActuator:
    def __init__(self, edges: list[dict], train_data, train: pd.DataFrame, info: dict):
        self.edges = edges; self.train_data = train_data; self.train = train
        self.numeric_names = [info["column_names"][i] for i in info["num_col_idx"]]
        self.categorical_names = [info["column_names"][i] for i in info["target_col_idx"] + info["cat_col_idx"]]
        self.num_pos = {name:i for i,name in enumerate(self.numeric_names)}
        self.cat_pos = {name:i for i,name in enumerate(self.categorical_names)}
        self.scales = {name:max(float(pd.to_numeric(train[name],errors="coerce").std()),1e-6) for name in self.numeric_names}
        self.compiled = []
        for edge in edges:
            if len(edge.get("requirements", [])) != 1: continue
            requirement = edge["requirements"][0]
            left = requirement["left_column"]
            if requirement["right_kind"] == "column":
                right = requirement["right_column"]
                if left in self.num_pos and right in self.num_pos and requirement["operator"] in {"<=",">=","<",">"}:
                    self.compiled.append({**edge,"kind":"numeric_column","requirement":requirement,
                        "columns":sorted(set(self._edge_columns(edge)))})
                continue
            if left in self.num_pos:
                raw = pd.to_numeric(train[left],errors="coerce").dropna().to_numpy(float)
                threshold=float(requirement["right_value"]); op=requirement["operator"]
                if op==">":
                    possible=raw[raw>threshold]; threshold=float(possible.min()) if len(possible) else float(np.nextafter(threshold,np.inf))
                elif op=="<":
                    possible=raw[raw<threshold]; threshold=float(possible.max()) if len(possible) else float(np.nextafter(threshold,-np.inf))
                self.compiled.append({**edge,"kind":"numeric_literal","requirement":requirement,
                    "target_raw":threshold,"columns":sorted(set(self._edge_columns(edge)))})
            elif left in self.cat_pos and requirement["operator"]=="==":
                categories=train_data.cat_transform.named_steps["ordinal"].categories_[self.cat_pos[left]]
                matches=np.flatnonzero(np.asarray(categories).astype(str)==str(requirement["right_value"]))
                if len(matches)==1:
                    self.compiled.append({**edge,"kind":"categorical_literal","requirement":requirement,
                        "target_code":int(matches[0]),"columns":sorted(set(self._edge_columns(edge)))})

    def _edge_columns(self, edge):
        atoms=edge.get("applicability",[])+edge.get("requirements",[])
        atoms += [a for g in edge.get("exceptions",[]) for a in g.get("all_of",[])]
        result=[]
        for atom in atoms:
            result.append(atom["left_column"])
            if atom.get("right_kind")=="column": result.append(atom["right_column"])
        return result

    def encode_numeric_targets(self, frame: pd.DataFrame, column: str, values: np.ndarray) -> np.ndarray:
        matrix=frame[self.numeric_names].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        matrix[:,self.num_pos[column]]=values
        return self.train_data.num_transform.transform(matrix)[:,self.num_pos[column]]

    def severity(self, frame: pd.DataFrame, rule: dict) -> np.ndarray:
        req=rule["requirement"]; left=pd.to_numeric(frame[req["left_column"]],errors="coerce").to_numpy(float)
        if req["right_kind"]=="column":
            right=pd.to_numeric(frame[req["right_column"]],errors="coerce").to_numpy(float)
            return np.abs(left-right)
        try: return np.abs(left-float(req["right_value"]))
        except Exception: return np.ones(len(frame))

    def project(self, numeric: torch.Tensor, categorical: torch.Tensor, preview: PreviewModel,
                frame: pd.DataFrame, rule: dict, rows: np.ndarray, strength: float) -> tuple[int,str]:
        if not rows.any(): return 0,"none"
        device=numeric.device; mask=torch.as_tensor(rows,device=device,dtype=torch.bool); req=rule["requirement"]
        amount=float(strength)*(1.0 if rule["classification"]=="hard" else .75)
        if rule["kind"]=="categorical_literal":
            categorical[mask,self.cat_pos[req["left_column"]]]=rule["target_code"]
            return int(mask.sum()),req["left_column"]
        if rule["kind"]=="numeric_literal":
            pos=self.num_pos[req["left_column"]]
            raw=np.full(len(frame),rule["target_raw"],dtype=float)
            encoded=torch.as_tensor(self.encode_numeric_targets(frame,req["left_column"],raw),device=device,dtype=numeric.dtype)
            current=numeric[mask,pos]; numeric[mask,pos]=current+amount*(encoded[mask]-current)
            return int(mask.sum()),req["left_column"]
        left,right=req["left_column"],req["right_column"]
        lpos,rpos=self.num_pos[left],self.num_pos[right]
        lraw=pd.to_numeric(frame[left],errors="coerce").to_numpy(float)
        rraw=pd.to_numeric(frame[right],errors="coerce").to_numpy(float)
        left_target=torch.as_tensor(self.encode_numeric_targets(frame,left,rraw),device=device,dtype=numeric.dtype)
        right_target=torch.as_tensor(self.encode_numeric_targets(frame,right,lraw),device=device,dtype=numeric.dtype)
        left_cost=(left_target-preview.numeric[:,lpos]).abs(); right_cost=(right_target-preview.numeric[:,rpos]).abs()
        protected={atom["left_column"] for atom in rule.get("applicability",[])}
        choose_left=left_cost.le(right_cost)
        if left in protected: choose_left[:]=False
        if right in protected: choose_left[:]=True
        left_rows=mask & choose_left; right_rows=mask & ~choose_left
        if left_rows.any():
            current=numeric[left_rows,lpos]; numeric[left_rows,lpos]=current+amount*(left_target[left_rows]-current)
        if right_rows.any():
            current=numeric[right_rows,rpos]; numeric[right_rows,rpos]=current+amount*(right_target[right_rows]-current)
        return int(mask.sum()),"mixed_minimal_projection"


def graph_metrics(frame: pd.DataFrame, reference: pd.DataFrame, edges: list[dict]) -> dict:
    rates=[]; heldout_errors=[]; calibration_errors=[]; total_support=total_bad=0
    for edge in edges:
        active,_,bad=edge_masks(frame,edge); ractive,_,rbad=edge_masks(reference,edge)
        if active.sum():
            rate=float(bad.sum()/active.sum()); rates.append(rate); total_support+=int(active.sum()); total_bad+=int(bad.sum())
            if ractive.sum(): heldout_errors.append(abs(rate-float(rbad.sum()/ractive.sum())))
            calibration_errors.append(abs(rate-float(edge["calibration"]["violation_rate"])))
    return {"macro_violation":float(np.mean(rates)) if rates else math.nan,
        "micro_violation":total_bad/total_support if total_support else math.nan,
        "heldout_violation_error":float(np.mean(heldout_errors)) if heldout_errors else math.nan,
        "calibration_violation_error":float(np.mean(calibration_errors)) if calibration_errors else math.nan}


def edge_metric_rows(frame: pd.DataFrame, reference: pd.DataFrame, edges: list[dict], method: str) -> list[dict]:
    rows=[]
    for edge in edges:
        active,_,bad=edge_masks(frame,edge); ractive,_,rbad=edge_masks(reference,edge)
        rate=float(bad.sum()/active.sum()) if active.sum() else math.nan
        heldout_rate=float(rbad.sum()/ractive.sum()) if ractive.sum() else math.nan
        calibration_rate=float(edge["calibration"]["violation_rate"])
        rows.append({"method":method,"candidate_id":edge["candidate_id"],
            "classification":edge["classification"],"usage_role":edge.get("usage_role","actuate"),"support":int(active.sum()),
            "violations":int(bad.sum()),"violation_rate":rate,
            "heldout_violation_rate":heldout_rate,"calibration_violation_rate":calibration_rate,
            "heldout_error":abs(rate-heldout_rate) if np.isfinite(rate) and np.isfinite(heldout_rate) else math.nan,
            "calibration_error":abs(rate-calibration_rate) if np.isfinite(rate) else math.nan})
    return rows


def evaluate_output(frame, base, heldout, edges, info, method, diagnostics):
    numeric=[info["column_names"][i] for i in info["num_col_idx"]]
    categorical=[info["column_names"][i] for i in info["cat_col_idx"]]
    ks=[ks_2samp(pd.to_numeric(frame[c]),pd.to_numeric(heldout[c])).statistic for c in numeric]
    tv=[]
    for c in categorical:
        a=frame[c].astype(str).value_counts(normalize=True); b=heldout[c].astype(str).value_counts(normalize=True)
        keys=a.index.union(b.index); tv.append(.5*float((a.reindex(keys,fill_value=0)-b.reindex(keys,fill_value=0)).abs().sum()))
    drift=np.mean([np.mean(np.abs(pd.to_numeric(frame[c]).to_numpy()-pd.to_numeric(base[c]).to_numpy()))/
        max(float(pd.to_numeric(heldout[c]).std()),1e-6) for c in numeric])
    return {"method":method,**graph_metrics(frame,heldout,edges),"numeric_ks":float(np.mean(ks)),
        "categorical_tv":float(np.mean(tv)),"numeric_drift_sd":float(drift),
        "action_events":diagnostics.get("action_events",0),
        "unique_rows_actuated":diagnostics.get("unique_rows_actuated",0),
        "edge_step_enabled":diagnostics.get("edge_step_enabled",0),
        "conflicts":diagnostics.get("conflicts",0),"runtime_seconds":diagnostics.get("runtime_seconds",0)}


@torch.no_grad()
def sample_guided(diffusion,preview,train_data,info,actuator,mode,samples,strength,min_support,drift,cutoffs):
    started=time.perf_counter(); device=diffusion.device
    times=torch.linspace(0,1,diffusion.num_timesteps,dtype=torch.float32,device=device)[:,None]
    sn=diffusion.num_schedule.total_noise(times); sc=diffusion.cat_schedule.total_noise(times)
    snx=torch.zeros_like(sn); snx[1:]=sn[:-1]; scx=torch.zeros_like(sc); scx[1:]=sc[:-1]
    numeric=torch.randn((samples,diffusion.num_numerical_features),device=device)*sn[-1]
    categorical=diffusion._sample_masked_prior(samples,len(diffusion.num_classes))
    previous_masks={}; previous_preview=None
    ever_acted=np.zeros(samples,dtype=bool)
    stats={"action_events":0,"unique_rows_actuated":0,"edge_step_enabled":0,"conflicts":0,
        "actions_by_edge":{},"trace":[]}
    for index in reversed(range(diffusion.num_timesteps)):
        numeric,categorical,_=diffusion.edm_update(numeric,categorical,index,times[index],times[index-1] if index>0 else None,times[index],sn[index],snx[index],sn[index],sc[index],scx[index],sc[index])
        t=float(times[index]); in_middle=.10<=t<=.50
        if mode=="fixed_middle" and not in_middle: continue
        if mode in {"edge_schedule","state_active","calibrated_state"} and not (.10<=t<=.75): continue
        encoded=torch.cat([preview.numeric,preview.codes.float()],dim=1); frame=decode(train_data,encoded,info)
        current_preview=preview.numeric.detach().cpu().numpy(); claimed=set(); step={"t":t,"edges":0,"actions":0}
        for rule in actuator.compiled:
            edge_id=rule["candidate_id"]
            if mode in {"edge_schedule","state_active","calibrated_state"} and t>cutoffs.get(edge_id,.25): continue
            active,_,violation=edge_masks(frame,rule)
            gate=violation.copy()
            if mode in {"state_active","calibrated_state"}:
                prior=previous_masks.get(edge_id,np.zeros(samples,dtype=bool))
                persistent=active & prior
                stable=np.ones(samples,dtype=bool)
                if previous_preview is None: stable[:]=False
                else:
                    for column in rule["columns"]:
                        if column in actuator.num_pos:
                            pos=actuator.num_pos[column]; stable &= np.abs(current_preview[:,pos]-previous_preview[:,pos])<=drift
                gate &= persistent & stable
                if int(active.sum())<min_support: gate[:]=False
            previous_masks[edge_id]=active
            if mode=="calibrated_state" and rule["classification"]=="near_hard" and gate.any():
                allowed=int(round(float(rule["calibration"]["violation_rate"])*int(active.sum())))
                excess=max(0,int(violation.sum())-allowed)
                indices=np.flatnonzero(gate)
                if excess<len(indices):
                    severity=actuator.severity(frame,rule)[indices]
                    keep=indices[np.argsort(-severity)[:excess]]; limited=np.zeros_like(gate); limited[keep]=True; gate=limited
            target=req_target(rule)
            if target in claimed and gate.any(): stats["conflicts"]+=int(gate.sum()); continue
            acted,_=actuator.project(numeric,categorical,preview,frame,rule,gate,strength)
            if acted:
                claimed.add(target); stats["action_events"]+=acted; stats["edge_step_enabled"]+=1
                stats["actions_by_edge"][edge_id]=stats["actions_by_edge"].get(edge_id,0)+acted
                ever_acted |= gate
                step["edges"]+=1; step["actions"]+=acted
        stats["trace"].append(step); previous_preview=current_preview
    stats["unique_rows_actuated"]=int(ever_acted.sum())
    stats["runtime_seconds"]=time.perf_counter()-started
    return torch.cat([numeric,categorical],dim=1).cpu(),stats


def req_target(rule):
    req=rule["requirement"]
    return req["left_column"] if req["right_kind"]=="literal" else tuple(sorted([req["left_column"],req["right_column"]]))


def final_repair(frame: pd.DataFrame, actuator: TypedActuator, calibrated: bool = False) -> tuple[pd.DataFrame,dict]:
    result=frame.copy(); changed=np.zeros(len(result),dtype=bool); events=0; by_edge={}
    for rule in actuator.compiled:
        _,_,bad=edge_masks(result,rule)
        if not bad.any(): continue
        if calibrated and rule["classification"] == "near_hard":
            active,_,_=edge_masks(result,rule)
            allowed=int(round(float(rule["calibration"]["violation_rate"])*int(active.sum())))
            excess=max(0,int(bad.sum())-allowed)
            indices=np.flatnonzero(bad)
            severity=actuator.severity(result,rule)[indices]
            selected=indices[np.argsort(-severity)[:excess]]
            limited=np.zeros_like(bad); limited[selected]=True; bad=limited
            if not bad.any(): continue
        req=rule["requirement"]
        if rule["kind"]=="numeric_literal": result.loc[bad,req["left_column"]]=rule["target_raw"]
        elif rule["kind"]=="categorical_literal": result.loc[bad,req["left_column"]]=req["right_value"]
        else:
            left,right=req["left_column"],req["right_column"]
            lcost=(pd.to_numeric(result[left])-pd.to_numeric(result[right])).abs()/actuator.scales[left]
            rcost=(pd.to_numeric(result[left])-pd.to_numeric(result[right])).abs()/actuator.scales[right]
            choose_left=bad & lcost.le(rcost); choose_right=bad & ~choose_left
            result.loc[choose_left,left]=result.loc[choose_left,right]
            result.loc[choose_right,right]=result.loc[choose_right,left]
        count=int(bad.sum()); events+=count; by_edge[rule["candidate_id"]]=count
        changed |= bad
    return result,{"action_events":events,"unique_rows_actuated":int(changed.sum()),"actions_by_edge":by_edge}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--dataset",default="shoppers")
    p.add_argument("--graph",type=Path,required=True); p.add_argument("--schedule",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--samples",type=int,default=4096)
    p.add_argument("--steps",type=int,default=50); p.add_argument("--strength",type=float,default=1.0)
    p.add_argument("--min-support",type=int,default=12); p.add_argument("--drift",type=float,default=.10)
    p.add_argument("--seed",type=int,default=20261005)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    seed_all(args.seed); sys.path.insert(0,str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model,UniModMLP
    info=json.loads((args.repo/"data"/args.dataset/"info.json").read_text())
    train_data,test_data=preprocess_dataset(args.repo,args.dataset)
    train,heldout=decode(train_data,train_data.X,info),decode(train_data,test_data.X,info)
    edges=json.loads(args.graph.read_text()); schedule=pd.read_csv(args.schedule)
    cutoffs={str(r.candidate_id):float(r.cutoff) for r in schedule.itertuples(index=False) if bool(r.readable) and np.isfinite(r.cutoff)}
    with args.config.open("rb") as h: config=pickle.load(h)
    params=copy.deepcopy(config["unimodmlp_params"]); params["d_numerical"]=train_data.d_numerical; params["categories"]=(train_data.categories+1).tolist()
    model=Model(UniModMLP(**params),**config["diffusion_params"]["edm_params"])
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False); model.load_state_dict(state["denoise_fn"],strict=True)
    preview=PreviewModel(model,train_data.categories)
    dp=copy.deepcopy(config["diffusion_params"]); dp.update(num_timesteps=args.steps,scheduler="power_mean_per_column",cat_scheduler="log_linear_per_column")
    dp["sampler_params"]=copy.deepcopy(dp["sampler_params"]); dp["sampler_params"].update(stochastic_sampler=False,second_order_correction=False)
    diffusion=UnifiedCtimeDiffusion(train_data.categories,train_data.d_numerical,preview,None,**dp,device=torch.device("cuda")).cuda()
    diffusion.num_schedule.load_state_dict(state["num_schedule"]); diffusion.cat_schedule.load_state_dict(state["cat_schedule"]); diffusion.eval()
    actuator=TypedActuator(edges,train_data,train,info)
    if len(actuator.compiled)!=len(edges): raise RuntimeError(f"only compiled {len(actuator.compiled)}/{len(edges)} edges")
    common_seed=args.seed+100000; seed_all(common_seed); base=decode(train_data,diffusion.sample(args.samples),info)
    rows=[evaluate_output(base,base,heldout,edges,info,"unconditional",{})]
    edge_rows=edge_metric_rows(base,heldout,edges,"unconditional")
    for mode in ["fixed_middle","edge_schedule","state_active","calibrated_state"]:
        seed_all(common_seed); encoded,diagnostics=sample_guided(diffusion,preview,train_data,info,actuator,mode,args.samples,args.strength,args.min_support,args.drift,cutoffs)
        frame=decode(train_data,encoded,info); rows.append(evaluate_output(frame,base,heldout,edges,info,mode,diagnostics))
        edge_rows.extend(edge_metric_rows(frame,heldout,edges,mode))
        (args.output/f"trace_{mode}.json").write_text(json.dumps(diagnostics,indent=2))
    repaired,repair_diagnostics=final_repair(base,actuator)
    rows.append(evaluate_output(repaired,base,heldout,edges,info,"final_repair",repair_diagnostics))
    edge_rows.extend(edge_metric_rows(repaired,heldout,edges,"final_repair"))
    calibrated_repair,calibrated_repair_diagnostics=final_repair(base,actuator,calibrated=True)
    rows.append(evaluate_output(calibrated_repair,base,heldout,edges,info,"calibrated_final_repair",calibrated_repair_diagnostics))
    edge_rows.extend(edge_metric_rows(calibrated_repair,heldout,edges,"calibrated_final_repair"))
    metrics=pd.DataFrame(rows); metrics.to_csv(args.output/"method_metrics.csv",index=False)
    pd.DataFrame(edge_rows).to_csv(args.output/"edge_metrics.csv",index=False)
    (args.output/"active_graph.json").write_text(json.dumps(edges,indent=2))
    lines=["# Typed dynamic-edge guidance", "", "| Method | Macro violation | Heldout error | Calibration error | Numeric KS | Categorical TV | Drift SD | Action events | Unique rows |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in metrics.itertuples(index=False): lines.append(f"| {r.method} | {r.macro_violation:.4f} | {r.heldout_violation_error:.4f} | {r.calibration_violation_error:.4f} | {r.numeric_ks:.4f} | {r.categorical_tv:.4f} | {r.numeric_drift_sd:.4f} | {r.action_events} | {r.unique_rows_actuated} |")
    (args.output/"RESULTS.md").write_text("\n".join(lines)+"\n"); print("\n".join(lines))


if __name__=="__main__": main()
