#!/usr/bin/env python3
"""Unconditional, best-of-N, and final-projection baselines cross-dataset."""
from __future__ import annotations
import argparse, copy, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cross_dataset_condition_adapter as xc
from conditioned_tabdiff_adapter import decode, seed_all
from tabdiff_probe import preprocess_dataset

def shortfall(frame, contract, scales):
 out=np.zeros(len(frame))
 for c,v in contract["predicates"]: out += np.maximum(v-pd.to_numeric(frame[c],errors="coerce").to_numpy(),0)/scales[c]
 return out
def select(pool,contract,n,width,scales):
 rows=[]
 for i in range(n):
  g=pool.iloc[i*width:(i+1)*width]; valid=xc.mask(g,contract).to_numpy(); k=int(np.flatnonzero(valid)[0]) if valid.any() else int(shortfall(g,contract,scales).argmin()); rows.append(g.iloc[k])
 return pd.DataFrame(rows).reset_index(drop=True)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--dataset",choices=["default","shoppers"],required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--samples",type=int,default=512); p.add_argument("--steps",type=int,default=30); p.add_argument("--seed",type=int,default=20260925); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 if not torch.cuda.is_available(): raise RuntimeError("GPU required")
 device=torch.device("cuda"); seed_all(a.seed); sys.path.insert(0,str(a.repo)); started=time.perf_counter()
 from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
 from tabdiff.modules.main_modules import Model,UniModMLP
 info=json.loads((a.repo/"data"/a.dataset/"info.json").read_text()); tr,_=preprocess_dataset(a.repo,a.dataset); real=decode(tr,tr.X,info); xc.CATALOGS_CURRENT=[info["column_names"][i] for i in info["num_col_idx"]]
 with a.config.open("rb") as h: cfg=pickle.load(h)
 params=copy.deepcopy(cfg["unimodmlp_params"]); params["d_numerical"]=tr.d_numerical; params["categories"]=(tr.categories+1).tolist(); model=Model(UniModMLP(**params),**cfg["diffusion_params"]["edm_params"])
 state=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model.load_state_dict(state["denoise_fn"])
 dp=copy.deepcopy(cfg["diffusion_params"]); dp.update(num_timesteps=a.steps,scheduler="power_mean_per_column",cat_scheduler="log_linear_per_column"); dp["sampler_params"]=copy.deepcopy(dp["sampler_params"]); dp["sampler_params"].update(stochastic_sampler=False,second_order_correction=False)
 diff=UnifiedCtimeDiffusion(tr.categories,tr.d_numerical,model,None,**dp,device=device).to(device); diff.num_schedule.load_state_dict(state["num_schedule"]); diff.cat_schedule.load_state_dict(state["cat_schedule"]); diff.eval()
 contracts=xc.make_catalog(a.dataset); scales={c:max(float(pd.to_numeric(real[c]).std()),1e-6) for c in xc.CATALOGS[a.dataset]["scales"]}; rows=[]
 for k,c in enumerate(contracts):
  seed_all(a.seed+10000+k)
  with torch.no_grad(): base=decode(tr,diff.sample(a.samples),info)
  br=float(xc.mask(base,c).mean()); m=xc.evaluate(base,real,c,br,br); m.update(dataset=a.dataset,method="unconditional",seed=a.seed,passes=1,edit_rate=0,target_distortion=0); rows.append(m)
  repaired=base.copy(); edit=np.zeros(len(base),bool); distort=[]
  for col,val in c["predicates"]:
   # Keep an immutable pre-edit copy. Without copy(), pandas may expose a
   # Series backed by the mutated frame and spuriously report zero distortion.
   old=pd.to_numeric(repaired[col]).copy(); bad=old.lt(val); repaired.loc[bad,col]=val; edit|=bad.to_numpy(); distort.append(float((pd.to_numeric(repaired[col])-old).abs().mean()/scales[col]))
  m=xc.evaluate(repaired,real,c,br,br); m.update(dataset=a.dataset,method="final_projection",seed=a.seed,passes=1,edit_rate=float(edit.mean()),target_distortion=float(np.mean(distort))); rows.append(m)
  for width in (2,4):
   seed_all(a.seed+20000+k*31+width)
   with torch.no_grad(): pool=decode(tr,diff.sample(a.samples*width),info)
   chosen=select(pool,c,a.samples,width,scales); m=xc.evaluate(chosen,real,c,br,br); m.update(dataset=a.dataset,method=f"best_of_{width}",seed=a.seed,passes=width,edit_rate=0,target_distortion=np.nan); rows.append(m)
 metrics=pd.DataFrame(rows); metrics.to_csv(a.output/"contract_metrics.csv",index=False); s=metrics.groupby(["method","split"]).agg(satisfaction=("satisfaction_rate","mean"),cohort_ks=("cohort_numeric_ks","mean"),categorical_tv=("cohort_categorical_tv","mean"),edit_rate=("edit_rate","mean"),target_distortion=("target_distortion","mean"),passes=("passes","mean")).reset_index(); s.to_csv(a.output/"summary.csv",index=False)
 meta={"status":"executed","scope":"proxy mechanisms, not paper reproductions","dataset":a.dataset,"seed":a.seed,"samples_per_contract":a.samples,"runtime_seconds":time.perf_counter()-started}; (a.output/"run_metadata.json").write_text(json.dumps(meta,indent=2)); print(s.to_string(index=False)); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
