#!/usr/bin/env python3
"""Sequential monitoring policies for a temporally evolving constraint graph."""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

PRED = re.compile(r"^\s*(.*?)\s*(==|>=|<=|>|<)\s*(.*?)\s*$")

def ordered(dataset, frame):
    if dataset == 'beijing':
        key=pd.to_datetime(dict(year=frame.year,month=frame.month,day=frame.day,hour=frame.hour),errors='coerce'); order=np.argsort(key.to_numpy())
    elif dataset == 'news': order=np.argsort(-pd.to_numeric(frame[' timedelta'],errors='coerce').to_numpy())
    elif dataset == 'shoppers':
        months={'Feb':2,'Mar':3,'May':5,'June':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}; order=np.argsort(frame.Month.astype(str).map(months).fillna(-1).to_numpy())
    else: raise ValueError(dataset)
    return frame.iloc[order].reset_index(drop=True)

def mask(frame, text):
    m=PRED.match(text)
    if not m: raise ValueError(text)
    column,op,literal=(x.strip() for x in m.groups())
    if column not in frame:
        matches=[c for c in frame if str(c).strip()==column]
        if len(matches)!=1: raise KeyError(column)
        column=matches[0]
    values=frame[column]
    if op=='==':
        numeric=pd.to_numeric(values,errors='coerce')
        try: threshold=float(literal); return numeric.eq(threshold).fillna(False).to_numpy()
        except ValueError: return values.astype(str).str.strip().eq(literal).to_numpy()
    numeric=pd.to_numeric(values,errors='coerce'); threshold=float(literal)
    return {'>':numeric.gt,'<':numeric.lt,'>=':numeric.ge,'<=':numeric.le}[op](threshold).fillna(False).to_numpy()

def rate(frame, edge):
    relation=str(edge).split('IF ',1)[1]; a,c=relation.split(' THEN ',1)
    am,cm=mask(frame,a),mask(frame,c); n=int(am.sum()); bad=int((am&~cm).sum())
    return n,bad,bad/n if n else math.nan

def wilson_low(bad,n,z=1.96):
    if n<=0:return math.nan
    p=bad/n; den=1+z*z/n; centre=(p+z*z/(2*n))/den; radius=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return max(0.,centre-radius)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--lifecycle',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--threshold',type=float,default=.05); p.add_argument('--bins',type=int,default=10); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.repo/'analysis/llm_feedback_20260922/scripts')); import cross_dataset_constraint_audit as audit
    discovered=pd.read_csv(a.lifecycle/'temporal_edges.csv'); discovered=discovered[discovered.frozen_pair_dedup_keep.astype(bool)].copy()
    trajectories=[]
    for dataset,edges in discovered.groupby('dataset'):
        tr,te,*_=audit.load_dataset(a.repo,dataset); full=ordered(dataset,pd.concat([tr,te],ignore_index=True)); bins=np.array_split(full,a.bins)
        for e in edges.itertuples(index=False):
            for b,part in enumerate(bins):
                n,bad,r=rate(part,e.edge); trajectories.append({'dataset':dataset,'edge':e.edge,'bin':b,'support':n,'bad':bad,'rate':r,'early_rate':e.early_rate})
    trace=pd.DataFrame(trajectories); trace.to_csv(a.output/'edge_bin_trajectory.csv',index=False)
    records=[]
    for (dataset,edge),g in trace.groupby(['dataset','edge']):
        g=g.sort_values('bin'); prior=[]; ewma=None; consecutive=0
        for row in g.itertuples(index=False):
            if row.bin < a.bins//2:
                prior.append((row.bad,row.support,row.rate)); continue
            stable=bool(np.isfinite(row.rate) and row.rate<=a.threshold and abs(row.rate-row.early_rate)<=a.threshold)
            last=prior[-1] if prior else (0,0,row.early_rate)
            if ewma is None: ewma=row.early_rate
            policies={
                'frozen':True,
                'last_window':bool(np.isfinite(last[2]) and last[2]<=a.threshold and abs(last[2]-row.early_rate)<=a.threshold),
                'ewma':bool(ewma<=a.threshold and abs(ewma-row.early_rate)<=a.threshold),
                'confidence_alarm':not bool(last[1] and wilson_low(last[0],last[1])>a.threshold),
                'two_alarm':consecutive<2,
                'oracle_current':stable,
            }
            for policy,active in policies.items(): records.append({'dataset':dataset,'edge':edge,'bin':row.bin,'policy':policy,'active':active,'stable_current':stable,'current_rate':row.rate,'support':row.support})
            failed=not stable; consecutive=consecutive+1 if failed else 0
            if np.isfinite(row.rate): ewma=.6*row.rate+.4*ewma
            prior.append((row.bad,row.support,row.rate))
    decisions=pd.DataFrame(records); decisions.to_csv(a.output/'sequential_decisions.csv',index=False)
    rows=[]
    for (dataset,policy),g in decisions.groupby(['dataset','policy']):
        safe=g.stable_current; active=g.active
        selected=g[active]
        finite=selected[np.isfinite(selected.current_rate)&selected.support.gt(0)]
        rows.append({'dataset':dataset,'policy':policy,'edge_windows':len(g),'active_fraction':active.mean(),'safe_retention':active[safe].mean() if safe.any() else math.nan,'unsafe_exposure':active[~safe].mean() if (~safe).any() else 0.,'unsafe_active_windows':int((active&~safe).sum()),'weighted_violation':float(np.average(finite.current_rate,weights=finite.support)) if len(finite) and finite.support.sum() else math.nan})
    out=pd.DataFrame(rows); out.to_csv(a.output/'summary.csv',index=False)
    lines=['# Sequential dynamic-graph monitoring','','Every decision for a monitoring window uses only earlier windows. `oracle_current` is an unattainable upper bound.','','| Dataset | Policy | Active | Safe retention | Unsafe exposure | Unsafe active windows | Weighted violation |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in out.itertuples(index=False): lines.append(f'| {r.dataset} | {r.policy} | {r.active_fraction:.4f} | {r.safe_retention:.4f} | {r.unsafe_exposure:.4f} | {r.unsafe_active_windows} | {r.weighted_violation:.4f} |')
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n'); (a.output/'run_metadata.json').write_text(json.dumps({'status':'executed','discovery':'first 50 percent','monitoring':'five sequential one-step-ahead windows','late_labels_used_for_decision':False,'semantic_claim':False},indent=2)); print(out.to_string(index=False))
if __name__=='__main__':main()
