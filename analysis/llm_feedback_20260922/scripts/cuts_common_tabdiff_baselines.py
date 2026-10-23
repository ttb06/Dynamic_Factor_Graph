#!/usr/bin/env python3
"""TabDiff baselines on the five public CuTS Adult constraint programs."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import ks_2samp
from conditioned_tabdiff_adapter import total_variation, seed_all
from tabdiff_probe import build_model, decode_tensor

def clean(f):
    f=f.copy()
    for c in f.select_dtypes(include=['object','string']).columns: f[c]=f[c].astype(str).str.strip()
    return f

def valid(f,n):
    if n=='implication2': return ~(f['marital.status'].isin(['Widowed']) | f['relationship'].eq('Wife')) | f['sex'].eq('Female')
    if n=='implication4': return ~f['marital.status'].isin(['Divorced','Never-married']) | ~f['relationship'].isin(['Husband','Wife'])
    if n=='implication5': return ~f['workclass'].isin(['Federal-gov','Local-gov','State-gov']) | f['education'].isin(['Bachelors','Some-college','Masters','Doctorate'])
    if n=='line_constraint1': return f['sex'].eq('Female')
    if n=='line_constraint2': return pd.to_numeric(f['age']).gt(35) & pd.to_numeric(f['age']).lt(55)
    raise KeyError(n)

def repair(f,n):
    out=f.copy(); bad=~valid(out,n)
    if n=='implication2': out.loc[bad,'sex']='Female'
    elif n=='implication4': out.loc[bad,'relationship']='Unmarried'
    elif n=='implication5': out.loc[bad,'education']='Bachelors'
    elif n=='line_constraint1': out.loc[bad,'sex']='Female'
    elif n=='line_constraint2': out.loc[pd.to_numeric(out.age).le(35),'age']=36; out.loc[pd.to_numeric(out.age).ge(55),'age']=54
    return out,bad.to_numpy()

def fidelity(f,real,num,cat):
    return float(np.mean([ks_2samp(pd.to_numeric(f[c]),pd.to_numeric(real[c])).statistic for c in num])), float(np.mean([total_variation(f[c],real[c]) for c in cat]))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--samples',type=int,default=4096); p.add_argument('--best-of',type=int,default=8); p.add_argument('--steps',type=int,default=30); p.add_argument('--seed',type=int,default=20260930); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda'); seed_all(a.seed); started=time.perf_counter(); b=build_model(a.repo,a.checkpoint,a.config,a.steps,device); real=clean(decode_tensor(b,b.dataset_test.X)); info=b.info
    num=[info['column_names'][i] for i in info['num_col_idx']]; cat=[info['column_names'][i] for i in info['cat_col_idx']+info['target_col_idx']]
    names=['implication2','implication4','implication5','line_constraint1','line_constraint2']; rows=[]
    for k,n in enumerate(names):
        seed_all(a.seed+k*100); base=clean(decode_tensor(b,b.diffusion.sample(a.samples)))
        methods={'unconditional':(base,np.zeros(len(base),bool),1)}
        fixed,ed=repair(base,n); methods['final_repair']=(fixed,ed,1)
        seed_all(a.seed+k*100+17); pool=clean(decode_tensor(b,b.diffusion.sample(a.samples*a.best_of)))
        chosen=[]
        for i in range(a.samples):
            g=pool.iloc[i*a.best_of:(i+1)*a.best_of]; ok=valid(g,n).to_numpy(); chosen.append(g.iloc[int(np.flatnonzero(ok)[0]) if ok.any() else 0])
        selected=pd.DataFrame(chosen).reset_index(drop=True); selected, fallback=repair(selected,n); methods[f'best_of_{a.best_of}_plus_fallback']=(selected,fallback,a.best_of)
        for m,(f,e,passes) in methods.items():
            ks,tv=fidelity(f,real,num,cat); rows.append({'program':n,'method':m,'satisfaction':float(valid(f,n).mean()),'numeric_ks':ks,'categorical_tv':tv,'edit_rate':float(e.mean()),'denoiser_passes':passes})
    df=pd.DataFrame(rows); df.to_csv(a.output/'metrics.csv',index=False)
    lines=['# TabDiff baselines on public CuTS Adult programs','','| Program | Method | Satisfaction | Numeric KS | Categorical TV | Edit rate | Passes |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in df.itertuples(index=False): lines.append(f'| {r.program} | {r.method} | {r.satisfaction:.4f} | {r.numeric_ks:.4f} | {r.categorical_tv:.4f} | {r.edit_rate:.4f} | {r.denoiser_passes} |')
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n'); (a.output/'run_metadata.json').write_text(json.dumps({'status':'executed','program_source':'official CuTS repository','generator':'official TabDiff checkpoint','wrappers':'internal repair and best-of selection','samples':a.samples,'best_of':a.best_of,'runtime_seconds':time.perf_counter()-started},indent=2)); print(df.to_string(index=False))
if __name__=='__main__': main()
