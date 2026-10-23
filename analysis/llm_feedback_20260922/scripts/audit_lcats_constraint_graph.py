#!/usr/bin/env python3
"""Audit official LCATS-discovered graph for support, novelty and heldout validity."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--lcats-root',type=Path,required=True); p.add_argument('--dataset',default='insurance'); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    d=a.lcats_root/'data'/a.dataset; constraints=json.loads((d/'constraints.json').read_text()); metadata=json.loads((d/'metadata.json').read_text()); train=pd.read_csv(d/'train.csv'); test=pd.read_csv(d/'test.csv')
    spec=importlib.util.spec_from_file_location('lcats_verify',d/'verify_script.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rows=[]
    for i,c in enumerate(constraints):
        tr_bad=mod.verify(train)[i]; te_bad=mod.verify(test)[i]; cols=c['cols']; redundant=False
        if c['type']=='domain_bound':
            formula=c['formula']; redundant=True
            for col in cols:
                if col not in metadata: redundant=False
                if "isin" in formula:
                    observed=set(train[col].astype(str)); declared=set(metadata[col].get('range',[])); redundant = redundant and (not declared or observed.issubset(declared))
        rows.append({'id':c['id'],'type':c['type'],'description':c['desc'],'columns':'|'.join(cols),'train_violations':int(tr_bad.sum()),'train_rate':float(tr_bad.mean()),'test_violations':int(te_bad.sum()),'test_rate':float(te_bad.mean()),'metadata_or_domain_constraint':bool(c['type']=='domain_bound'),'likely_schema_redundant':redundant})
    f=pd.DataFrame(rows); f.to_csv(a.output/'constraint_audit.csv',index=False)
    summary={'constraints':len(f),'domain_bounds':int((f.type=='domain_bound').sum()),'relational_or_conditional':int(f.type.isin(['order','conditional']).sum()),'zero_train_violations':int((f.train_violations==0).sum()),'zero_test_violations':int((f.test_violations==0).sum()),'nonzero_heldout_violations':int((f.test_violations>0).sum())}
    lines=['# Audit of official LCATS-discovered constraint graph','',f"Summary: `{summary}`",'','| ID | Type | Constraint | Train violation | Test violation |','| ---: | --- | --- | ---: | ---: |']
    for r in f.itertuples(index=False): lines.append(f'| {r.id} | {r.type} | {r.description} | {r.train_rate:.4f} ({r.train_violations}) | {r.test_rate:.4f} ({r.test_violations}) |')
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n'); (a.output/'run_summary.json').write_text(json.dumps(summary,indent=2)); print(f.to_string(index=False)); print(summary)
if __name__=='__main__': main()
