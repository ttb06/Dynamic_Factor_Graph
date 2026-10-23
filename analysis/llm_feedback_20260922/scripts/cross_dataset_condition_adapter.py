#!/usr/bin/env python3
"""Typed/request-ID conditional TabDiff adapters on Default and Shoppers.

Contracts are controlled cohort requests, not discovered domain invariants.
They test held-out numerical parameters and unseen conjunctions while reusing
the dataset-specific pretrained checkpoints.
"""
from __future__ import annotations
import argparse, copy, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader, TensorDataset

from conditioned_tabdiff_adapter import ConditionedUniModMLP, ContractConditionedModel, decode, seed_all, total_variation
from tabdiff_probe import preprocess_dataset


CATALOGS = {
 "adult": {
  "scales":{"age":100.0,"hours.per.week":100.0,"education.num":20.0},
  "train":{"age":[25,35,50],"hours.per.week":[20,40,50],"education.num":[9,10,13]},
  "heldout":{"age":[30,45],"hours.per.week":[30,45],"education.num":[11,12]},
  "composition":[[("age",35),("hours.per.week",40)],[("hours.per.week",30),("education.num",12)],[("age",45),("education.num",10)]],
 },
 "default": {
  "scales":{"LIMIT_BAL":1_000_000.0,"AGE":100.0,"BILL_AMT1":1_000_000.0},
  "train":{"LIMIT_BAL":[50_000,140_000,360_000],"AGE":[25,34,49],"BILL_AMT1":[0,22_000,140_000]},
  "heldout":{"LIMIT_BAL":[100_000,250_000],"AGE":[30,45],"BILL_AMT1":[10_000,70_000]},
  "composition":[[("LIMIT_BAL",140_000),("AGE",34)],[("AGE",30),("BILL_AMT1",70_000)],[("LIMIT_BAL",250_000),("BILL_AMT1",10_000)]],
 },
 "shoppers": {
  "scales":{"PageValues":400.0,"Administrative_Duration":3_500.0,"ProductRelated_Duration":65_000.0},
  "train":{"PageValues":[1,10,40],"Administrative_Duration":[10,90,220],"ProductRelated_Duration":[200,600,2_800]},
  "heldout":{"PageValues":[5,25],"Administrative_Duration":[50,150],"ProductRelated_Duration":[400,1_500]},
  "composition":[[("PageValues",10),("ProductRelated_Duration",600)],[("Administrative_Duration",50),("ProductRelated_Duration",1_500)],[("PageValues",5),("Administrative_Duration",150)]],
 },
}

def make_catalog(dataset):
 spec=CATALOGS[dataset]; out=[]
 for split,key in [("train","train"),("heldout_threshold","heldout")]:
  for col,vals in spec[key].items():
   for val in vals:
    out.append({"contract_id":f"{col}_ge_{val:g}","split":split,"predicates":[(col,float(val))],"family":col})
 for n,preds in enumerate(spec["composition"]):
  out.append({"contract_id":"and_"+"_".join(f"{c}_ge_{v:g}" for c,v in preds),"split":"heldout_composition","predicates":[(c,float(v)) for c,v in preds],"family":f"composition_{n}"})
 return out

def vector(contract,dataset):
 by=dict(contract["predicates"]); out=[]
 for col,scale in CATALOGS[dataset]["scales"].items(): out += [float(col in by), float(by.get(col,0))/scale]
 return out

def mask(frame,contract):
 m=pd.Series(True,index=frame.index)
 for c,v in contract["predicates"]: m &= pd.to_numeric(frame[c],errors="coerce").ge(v)
 return m

def evaluate(sample, real, contract, frozen_rate, null_rate):
 sat=mask(sample,contract); ref=real[mask(real,contract)]
 nums=list(CATALOGS_CURRENT)
 cats=[c for c in real.columns if c not in nums]
 ks=[]
 for c in nums:
  if len(ref): ks.append(float(ks_2samp(pd.to_numeric(sample[c]),pd.to_numeric(ref[c])).statistic))
 tv=[total_variation(sample[c],ref[c]) for c in cats if len(ref)]
 return {"contract_id":contract["contract_id"],"split":contract["split"],"family":contract["family"],
         "satisfaction_rate":float(sat.mean()),"gain_frozen":float(sat.mean()-frozen_rate),"gain_null":float(sat.mean()-null_rate),
         "cohort_numeric_ks":float(np.mean(ks)),"cohort_categorical_tv":float(np.mean(tv)),"reference_cohort_size":len(ref)}

CATALOGS_CURRENT=[]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--dataset",choices=["adult","default","shoppers"],required=True)
 p.add_argument("--config",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
 p.add_argument("--representation",choices=["request_id","structured"],required=True); p.add_argument("--epochs",type=int,default=100)
 p.add_argument("--pairs-per-contract",type=int,default=2048); p.add_argument("--batch-size",type=int,default=2048); p.add_argument("--eval-samples",type=int,default=512)
 p.add_argument("--guidance-scales",default="1:2:4"); p.add_argument("--steps",type=int,default=30); p.add_argument("--seed",type=int,default=20260925)
 a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 if not torch.cuda.is_available(): raise RuntimeError("GPU required")
 device=torch.device("cuda"); seed_all(a.seed); sys.path.insert(0,str(a.repo)); started=time.perf_counter()
 from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
 from tabdiff.modules.main_modules import Model
 contracts=make_catalog(a.dataset); train_contracts=[x for x in contracts if x["split"]=="train"]
 train_ids=[x["contract_id"] for x in train_contracts]
 vectors={}
 for c in contracts:
  if a.representation=="structured": vectors[c["contract_id"]]=vector(c,a.dataset)
  else:
   v=[0.0]*len(train_ids)
   if c["contract_id"] in train_ids: v[train_ids.index(c["contract_id"])]=1.0
   vectors[c["contract_id"]]=v
 condition_dim=len(next(iter(vectors.values())))
 info=json.loads((a.repo/"data"/a.dataset/"info.json").read_text()); train_data,_=preprocess_dataset(a.repo,a.dataset)
 real=decode(train_data,train_data.X,info)
 global CATALOGS_CURRENT; CATALOGS_CURRENT=[info["column_names"][i] for i in info["num_col_idx"]]
 with a.config.open("rb") as h: config=pickle.load(h)
 params=copy.deepcopy(config["unimodmlp_params"]); params["d_numerical"]=train_data.d_numerical; params["categories"]=(train_data.categories+1).tolist()
 backbone=ConditionedUniModMLP(condition_dim=condition_dim,**params); base=Model(backbone,**config["diffusion_params"]["edm_params"])
 state=torch.load(a.checkpoint,map_location="cpu",weights_only=False); inc=base.load_state_dict(state["denoise_fn"],strict=False)
 missing=[x for x in inc.missing_keys if "condition_projection" not in x]
 if inc.unexpected_keys or missing: raise RuntimeError(str(inc))
 model=ContractConditionedModel(base,backbone,condition_dim).to(device)
 dp=copy.deepcopy(config["diffusion_params"]); dp.update(num_timesteps=a.steps,scheduler="power_mean_per_column",cat_scheduler="log_linear_per_column")
 dp["sampler_params"]=copy.deepcopy(dp["sampler_params"]); dp["sampler_params"].update(stochastic_sampler=False,second_order_correction=False)
 diff=UnifiedCtimeDiffusion(train_data.categories,train_data.d_numerical,model,None,**dp,device=device).to(device)
 diff.num_schedule.load_state_dict(state["num_schedule"]); diff.cat_schedule.load_state_dict(state["cat_schedule"])
 for q in diff.parameters(): q.requires_grad_(False)
 for q in backbone.mlp.condition_projection.parameters(): q.requires_grad_(True)
 for q in backbone.mlp.mlp.parameters(): q.requires_grad_(True)
 model.set_condition(torch.zeros(a.eval_samples,condition_dim,device=device),1); seed_all(a.seed+9000)
 with torch.no_grad(): frozen=decode(train_data,diff.sample(a.eval_samples),info)
 frozen_rates={c["contract_id"]:float(mask(frozen,c).mean()) for c in contracts}
 rng=np.random.default_rng(a.seed); xs=[]; cs=[]; counts={}
 for c in train_contracts:
  eligible=np.flatnonzero(mask(real,c).to_numpy())
  if not len(eligible): raise RuntimeError(f"no eligible rows {c['contract_id']}")
  idx=rng.choice(eligible,a.pairs_per_contract,replace=len(eligible)<a.pairs_per_contract)
  xs.append(train_data.X[idx]); cs.append(torch.tensor(vectors[c["contract_id"]],dtype=torch.float32).repeat(a.pairs_per_contract,1)); counts[c["contract_id"]]=len(eligible)
 idx=rng.choice(len(real),a.pairs_per_contract,replace=False); xs.append(train_data.X[idx]); cs.append(torch.zeros(a.pairs_per_contract,condition_dim))
 X,C=torch.cat(xs),torch.cat(cs); loader=DataLoader(TensorDataset(X,C),batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),pin_memory=True)
 opt=torch.optim.AdamW([{"params":backbone.mlp.condition_projection.parameters(),"lr":5e-4},{"params":backbone.mlp.mlp.parameters(),"lr":1e-5}],weight_decay=0)
 curve=[]
 for epoch in range(1,a.epochs+1):
  diff.train(); ds=csum=n=0.0
  for batch,cond in loader:
   batch,cond=batch.to(device),cond.to(device); drop=torch.rand(len(cond),device=device)<.1; cond=cond.clone(); cond[drop]=0
   model.set_condition(cond,1); opt.zero_grad(set_to_none=True); dl,cl=diff.mixed_loss(batch); (dl+cl).backward(); opt.step()
   ds+=float(dl.detach())*len(batch); csum+=float(cl.detach())*len(batch); n+=len(batch)
  curve.append({"epoch":epoch,"discrete_loss":ds/n,"continuous_loss":csum/n})
 pd.DataFrame(curve).to_csv(a.output/"training_curve.csv",index=False)
 diff.eval(); model.set_condition(torch.zeros(a.eval_samples,condition_dim,device=device),1); seed_all(a.seed+9000)
 with torch.no_grad(): null=decode(train_data,diff.sample(a.eval_samples),info)
 null_rates={c["contract_id"]:float(mask(null,c).mean()) for c in contracts}; rows=[]
 for g in [float(x) for x in a.guidance_scales.replace(",",":").split(":") if x]:
  for k,c in enumerate(contracts):
   cond=torch.tensor(vectors[c["contract_id"]],device=device).float().repeat(a.eval_samples,1); model.set_condition(cond,g); seed_all(a.seed+10000+k)
   with torch.no_grad(): sample=decode(train_data,diff.sample(a.eval_samples),info)
   r=evaluate(sample,real,c,frozen_rates[c["contract_id"]],null_rates[c["contract_id"]]); r.update(dataset=a.dataset,representation=a.representation,guidance_scale=g,seed=a.seed); rows.append(r)
 metrics=pd.DataFrame(rows); metrics.to_csv(a.output/"contract_metrics.csv",index=False)
 torch.save({"condition_projection":backbone.mlp.condition_projection.state_dict(),"latent_mlp":backbone.mlp.mlp.state_dict(),"vectors":vectors,"condition_dim":condition_dim},a.output/"adapter.pt")
 meta={"status":"executed","dataset":a.dataset,"representation":a.representation,"seed":a.seed,"epochs":a.epochs,"training_pairs":len(X),"eligible_counts":counts,"runtime_seconds":time.perf_counter()-started,"checkpoint":str(a.checkpoint),"contract_scope":"controlled cohort requests; not domain invariants"}
 (a.output/"run_metadata.json").write_text(json.dumps(meta,indent=2)); print(metrics.groupby(["guidance_scale","split"])[["satisfaction_rate","gain_null","cohort_numeric_ks"]].mean().to_string()); print(json.dumps(meta,indent=2))

if __name__=="__main__": main()
