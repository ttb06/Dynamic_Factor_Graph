#!/usr/bin/env python3
"""Paired-noise active-set controls on Default and Online Shoppers.

Policies are controlled generation requests, not asserted domain truths. A
first unconditional sample identifies row applicability; a second trajectory
uses the same initial noise and applies the condition only to selected rows.
Only the numeric consequent delta is actuated.
"""
from __future__ import annotations
import argparse, copy, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

from conditioned_tabdiff_adapter import ConditionedUniModMLP, decode, seed_all, total_variation
from tabdiff_probe import preprocess_dataset


POLICIES={
 "default":[
  {"id":"marriage_limit","antecedent":[("MARRIAGE","eq",["1"])],"exception":[("EDUCATION","eq",["3"])],"target":"LIMIT_BAL","threshold":100000.0,"condition_id":"LIMIT_BAL_ge_100000"},
  {"id":"education_age","antecedent":[("EDUCATION","in",["1","2"])],"exception":[("PAY_0","eq",["-2"])],"target":"AGE","threshold":30.0,"condition_id":"AGE_ge_30"},
  {"id":"delay_bill","antecedent":[("PAY_0","in",["2","3"])],"exception":[("MARRIAGE","eq",["0"])],"target":"BILL_AMT1","threshold":10000.0,"condition_id":"BILL_AMT1_ge_10000"},
 ],
 "shoppers":[
  {"id":"new_visitor_value","antecedent":[("VisitorType","eq",["New_Visitor"])],"exception":[("Weekend","eq",["True"])],"target":"PageValues","threshold":5.0,"condition_id":"PageValues_ge_5"},
  {"id":"weekend_admin","antecedent":[("Weekend","eq",["True"])],"exception":[("Month","eq",["Feb"])],"target":"Administrative_Duration","threshold":50.0,"condition_id":"Administrative_Duration_ge_50"},
  {"id":"holiday_product","antecedent":[("Month","in",["Nov","Dec"])],"exception":[("TrafficType","eq",["13"])],"target":"ProductRelated_Duration","threshold":400.0,"condition_id":"ProductRelated_Duration_ge_400"},
 ],
}


def clause(frame: pd.DataFrame, predicates: list[tuple]) -> pd.Series:
    result=pd.Series(True,index=frame.index)
    for column,operator,values in predicates:
        observed=frame[column].astype(str).str.strip()
        result &= observed.eq(values[0]) if operator=="eq" else observed.isin(values)
    return result


def applicability(frame: pd.DataFrame, policy: dict) -> pd.Series:
    return clause(frame,policy["antecedent"]) & ~clause(frame,policy["exception"])


class TargetExternalGateModel(torch.nn.Module):
    def __init__(self,base,backbone,condition_dim):
        super().__init__(); self.base=base; self.backbone=backbone; self.condition_dim=condition_dim
        self.condition=None; self.gate=None; self.guidance=1.0; self.target_index=0
    def set_request(self,condition,gate,guidance,target_index):
        self.condition=condition; self.gate=gate; self.guidance=float(guidance); self.target_index=int(target_index)
    def forward(self,x_num,x_cat,t,sigma=None):
        condition=self.condition
        if condition is None: condition=torch.zeros(len(x_num),self.condition_dim,device=x_num.device,dtype=x_num.dtype)
        gate=self.gate
        if gate is None: gate=torch.zeros(len(x_num),device=x_num.device,dtype=x_num.dtype)
        self.backbone.set_condition(torch.zeros_like(condition)); base_num,base_cat=self.base(x_num,x_cat,t,sigma=sigma)
        self.backbone.set_condition(condition*gate.reshape(-1,1)); active_num,active_cat=self.base(x_num,x_cat,t,sigma=sigma)
        numeric=base_num.clone(); numeric[:,self.target_index]=base_num[:,self.target_index]+self.guidance*(active_num[:,self.target_index]-base_num[:,self.target_index])
        return numeric,base_cat


def repair(frame: pd.DataFrame, active: pd.Series, policy: dict) -> tuple[pd.DataFrame,float]:
    result=frame.copy(); values=pd.to_numeric(result[policy["target"]]); bad=active & values.lt(policy["threshold"])
    result.loc[bad,policy["target"]]=policy["threshold"]
    return result,float(bad.mean())


def evaluate(frame: pd.DataFrame,base: pd.DataFrame,real: pd.DataFrame,base_active: pd.Series,base_exception: pd.Series,policy: dict,method: str,seed: int,guidance: float,edit_rate: float=0.0) -> dict:
    target=policy["target"]; scale=max(float(pd.to_numeric(real[target]).std()),1e-6); values=pd.to_numeric(frame[target]); base_values=pd.to_numeric(base[target])
    violation=base_active & values.lt(policy["threshold"]); active_count=max(int(base_active.sum()),1)
    numeric=[column for column in NUMERIC_COLUMNS]; categorical=[column for column in frame.columns if column not in numeric]
    final_active=applicability(frame,policy)
    return {"dataset":DATASET,"policy_id":policy["id"],"method":method,"seed":seed,"guidance":guidance,"base_active_rate":float(base_active.mean()),
            "conditional_violation_rate":float(violation.sum()/active_count),"applicability_flip_rate":float((final_active!=base_active).mean()),
            "inactive_target_drift":float((values[~base_active]-base_values[~base_active]).abs().mean()/scale) if (~base_active).any() else 0.0,
            "active_target_drift":float((values[base_active]-base_values[base_active]).abs().mean()/scale) if base_active.any() else 0.0,
            "exception_target_drift":float((values[base_exception]-base_values[base_exception]).abs().mean()/scale) if base_exception.any() else 0.0,
            "numeric_ks_to_base":float(np.mean([ks_2samp(pd.to_numeric(frame[c]),pd.to_numeric(base[c])).statistic for c in numeric])),
            "categorical_tv_to_base":float(np.mean([total_variation(frame[c],base[c]) for c in categorical])),
            "edit_rate":edit_rate}


NUMERIC_COLUMNS=[]; DATASET=""
def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",type=Path,required=True); parser.add_argument("--dataset",choices=["default","shoppers"],required=True)
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--adapter",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=2048); parser.add_argument("--steps",type=int,default=30); parser.add_argument("--guidance",type=float,required=True); parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    global DATASET,NUMERIC_COLUMNS; DATASET=args.dataset; device=torch.device("cuda"); seed_all(args.seed); started=time.perf_counter(); sys.path.insert(0,str(args.repo))
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model
    info=json.loads((args.repo/"data"/args.dataset/"info.json").read_text()); train_data,_=preprocess_dataset(args.repo,args.dataset); real=decode(train_data,train_data.X,info)
    NUMERIC_COLUMNS=[info["column_names"][index] for index in info["num_col_idx"]]; numeric_positions={name:index for index,name in enumerate(NUMERIC_COLUMNS)}
    with args.config.open("rb") as handle: config=pickle.load(handle)
    saved=torch.load(args.adapter,map_location="cpu",weights_only=False); condition_dim=int(saved["condition_dim"]); vectors=saved["vectors"]
    params=copy.deepcopy(config["unimodmlp_params"]); params["d_numerical"]=train_data.d_numerical; params["categories"]=(train_data.categories+1).tolist()
    backbone=ConditionedUniModMLP(condition_dim=condition_dim,**params); base_model=Model(backbone,**config["diffusion_params"]["edm_params"])
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False); incompatible=base_model.load_state_dict(checkpoint["denoise_fn"],strict=False)
    missing=[key for key in incompatible.missing_keys if "condition_projection" not in key]
    if incompatible.unexpected_keys or missing: raise RuntimeError(str(incompatible))
    backbone.mlp.condition_projection.load_state_dict(saved["condition_projection"]); backbone.mlp.mlp.load_state_dict(saved["latent_mlp"])
    model=TargetExternalGateModel(base_model,backbone,condition_dim).to(device)
    dp=copy.deepcopy(config["diffusion_params"]); dp.update(num_timesteps=args.steps,scheduler="power_mean_per_column",cat_scheduler="log_linear_per_column")
    dp["sampler_params"]=copy.deepcopy(dp["sampler_params"]); dp["sampler_params"].update(stochastic_sampler=False,second_order_correction=False)
    diffusion=UnifiedCtimeDiffusion(train_data.categories,train_data.d_numerical,model,None,**dp,device=device).to(device); diffusion.num_schedule.load_state_dict(checkpoint["num_schedule"]); diffusion.cat_schedule.load_state_dict(checkpoint["cat_schedule"]); diffusion.eval()
    rows=[]
    for ordinal,policy in enumerate(POLICIES[args.dataset]):
        condition=torch.tensor(vectors[policy["condition_id"]],dtype=torch.float32,device=device).repeat(args.samples,1); zero=torch.zeros(args.samples,device=device)
        common_seed=args.seed+10000+ordinal; model.set_request(condition,zero,args.guidance,numeric_positions[policy["target"]]); seed_all(common_seed)
        with torch.no_grad(): base=decode(train_data,diffusion.sample(args.samples),info)
        base_active=applicability(base,policy); base_exception=clause(base,policy["antecedent"]) & clause(base,policy["exception"])
        rows.append(evaluate(base,base,real,base_active,base_exception,policy,"no_actuation",args.seed,args.guidance))
        base_values=pd.to_numeric(base[policy["target"]],errors="coerce")
        base_violated=base_active & base_values.lt(policy["threshold"])
        gates={
            "global_condition":torch.ones(args.samples,device=device),
            "preview_selective":torch.tensor(base_active.to_numpy(),dtype=torch.float32,device=device),
            "preview_violation_selective":torch.tensor(base_violated.to_numpy(),dtype=torch.float32,device=device),
            "shuffled_selective":torch.tensor(np.roll(base_active.to_numpy(),1),dtype=torch.float32,device=device),
        }
        preview_frames={}
        for method,gate in gates.items():
            model.set_request(condition,gate,args.guidance,numeric_positions[policy["target"]]); seed_all(common_seed)
            with torch.no_grad(): frame=decode(train_data,diffusion.sample(args.samples),info)
            rows.append(evaluate(frame,base,real,base_active,base_exception,policy,method,args.seed,args.guidance))
            if method in {"preview_selective","preview_violation_selective"}: preview_frames[method]=frame
        repaired,rate=repair(base,base_active,policy); rows.append(evaluate(repaired,base,real,base_active,base_exception,policy,"selective_postrepair",args.seed,args.guidance,rate))
        certified,rate=repair(preview_frames["preview_selective"],base_active,policy); rows.append(evaluate(certified,base,real,base_active,base_exception,policy,"preview_then_certifier",args.seed,args.guidance,rate))
        certified,rate=repair(preview_frames["preview_violation_selective"],base_active,policy); rows.append(evaluate(certified,base,real,base_active,base_exception,policy,"preview_violation_then_certifier",args.seed,args.guidance,rate))
    metrics=pd.DataFrame(rows); baseline=metrics[metrics.method.eq("no_actuation")][["policy_id","conditional_violation_rate"]].rename(columns={"conditional_violation_rate":"baseline_violation"}); metrics=metrics.merge(baseline,on="policy_id"); metrics["violation_reduction"]=metrics.baseline_violation-metrics.conditional_violation_rate; metrics.to_csv(args.output/"active_set_metrics.csv",index=False)
    summary=metrics.groupby("method").agg(policies=("policy_id","size"),violation=("conditional_violation_rate","mean"),reduction=("violation_reduction","mean"),inactive_drift=("inactive_target_drift","mean"),active_drift=("active_target_drift","mean"),exception_drift=("exception_target_drift","mean"),applicability_flip=("applicability_flip_rate","mean"),numeric_ks=("numeric_ks_to_base","mean"),categorical_tv=("categorical_tv_to_base","mean"),edit_rate=("edit_rate","mean")).reset_index(); summary.to_csv(args.output/"summary.csv",index=False)
    metadata={"status":"executed","dataset":args.dataset,"seed":args.seed,"guidance":args.guidance,"samples_per_policy":args.samples,"steps":args.steps,"scope":"controlled row-conditional policies; paired-noise two-pass active-set diagnostic","runtime_seconds":time.perf_counter()-started}
    (args.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2)); print(summary.to_string(index=False)); print(json.dumps(metadata,indent=2),flush=True)

if __name__=="__main__": main()
