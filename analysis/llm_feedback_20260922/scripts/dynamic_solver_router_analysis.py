#!/usr/bin/env python3
"""Pareto and leave-one-policy-out solver-router analysis."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

METRICS=['active_drift_sd','active_target_ks_reference','active_target_wasserstein_sd','active_cohort_numeric_ks','active_cohort_categorical_tv','target_context_spearman_mae']
PROFILES={
 'balanced':np.ones(6),
 'intervention_conservative':np.array([3,1,1,1,1,1]),
 'target_fidelity':np.array([1,3,3,1,1,1]),
 'joint_fidelity':np.array([1,1,1,3,3,3]),
}
KEEP=['subgroup_final_repair','empirical_marginal_bad_postrepair','empirical_stratified_bad_postrepair','diffusion_impute_then_certify','diffusion_impute_numgraph_then_certify','diffusion_bestof8_target_then_certify','diffusion_bestof8_numgraph_then_certify','lookahead_anchored_violated_masked_t100_plus_certifier','lookahead_anchored_violated_topk3_t100_plus_certifier','lookahead_anchored_violated_topk5_t100_plus_certifier','lookahead_anchored_violated_numgraph_t100_plus_certifier','lookahead_anchored_violated_t100_plus_certifier']

def pareto(group):
    x=group[METRICS].to_numpy(float); x=np.where(np.isfinite(x),x,np.inf); keep=[]
    for i in range(len(x)): keep.append(not np.any(np.all(x<=x[i],axis=1)&np.any(x<x[i],axis=1)))
    return np.array(keep)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--default',type=Path,required=True); p.add_argument('--shoppers',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    frames=[]
    for ds,path in [('default',a.default),('shoppers',a.shoppers)]:
        f=pd.read_csv(path/'policy_metrics.csv'); f['dataset']=ds; frames.append(f)
    raw=pd.concat(frames,ignore_index=True); raw=raw[raw.method.isin(KEEP)&(raw.active_satisfaction>=.999)&(raw.exception_changed.fillna(0)<=1e-9)].copy(); raw['policy_key']=raw.dataset+':'+raw.policy
    rows=[]; features=[]
    for key,g in raw.groupby('policy_key'):
        if len(g)<3: continue
        ranks=g[METRICS].rank(method='average',pct=True).fillna(1.0)
        pdmask=pareto(g); g=g.reset_index(drop=True); ranks=ranks.reset_index(drop=True)
        for profile,w in PROFILES.items():
            score=(ranks.to_numpy()*w).sum(1)/w.sum()
            for i,r in g.iterrows(): rows.append({'policy_key':key,'dataset':r.dataset,'policy':r.policy,'regime':r.regime,'threshold':r.threshold,'method':r.method,'profile':profile,'score':float(score[i]),'pareto':bool(pdmask[i])})
        u=raw[(raw.policy_key==key)&(raw.method=='subgroup_final_repair')]
        base=frames[0].iloc[0:0]
        allrow=pd.concat(frames); unc=allrow[(allrow.dataset.eq(g.dataset.iloc[0]) if 'dataset' in allrow else False)] if False else None
        # Policy features derive only from the unconditional row and request metadata.
        src=pd.concat(frames,ignore_index=True); q=src[(src.dataset==g.dataset.iloc[0])&(src.policy==g.policy.iloc[0])&(src.method=='unconditional')].iloc[0]
        features.append({'policy_key':key,'dataset':g.dataset.iloc[0],'regime':g.regime.iloc[0],'threshold':float(g.threshold.iloc[0]),'difficulty':1-float(q.active_satisfaction),'active_n':float(q.active_n),'exception_n':float(q.exception_n)})
    scores=pd.DataFrame(rows); scores.to_csv(a.output/'policy_method_scores.csv',index=False); feat=pd.DataFrame(features).drop_duplicates('policy_key')
    summary=[]; routing=[]
    for profile,s in scores.groupby('profile'):
        mean=s.groupby('method').score.mean().sort_values(); fixed=mean.index[0]; fixed_score=float(mean.iloc[0]); oracle=s.loc[s.groupby('policy_key').score.idxmin()]; oracle_score=float(oracle.score.mean())
        # Leave-one-policy-out shallow graph-feature router.
        chosen=[]
        labels=oracle.set_index('policy_key').method
        data=feat[feat.policy_key.isin(labels.index)].copy(); data['label']=data.policy_key.map(labels)
        for i,row in data.iterrows():
            tr=data.index!=i; numeric=['threshold','difficulty','active_n','exception_n']; categorical=['dataset','regime']
            pipe=Pipeline([('prep',ColumnTransformer([('num',StandardScaler(),numeric),('cat',OneHotEncoder(handle_unknown='ignore'),categorical)])),('tree',DecisionTreeClassifier(max_depth=3,min_samples_leaf=2,class_weight='balanced',random_state=0))])
            pipe.fit(data.loc[tr,numeric+categorical],data.loc[tr,'label']); pred=pipe.predict(data.loc[[i],numeric+categorical])[0]
            options=s[s.policy_key==row.policy_key]; hit=options[options.method==pred]
            if hit.empty: pred=fixed; hit=options[options.method==pred]
            chosen.append({'profile':profile,'policy_key':row.policy_key,'oracle_method':row.label,'chosen_method':pred,'chosen_score':float(hit.score.iloc[0]),'oracle_score':float(options.score.min()),'fixed_score':float(options[options.method==fixed].score.iloc[0])})
        c=pd.DataFrame(chosen); routing.extend(chosen); summary.append({'profile':profile,'policies':len(oracle),'best_fixed_method':fixed,'best_fixed_score':fixed_score,'oracle_router_score':oracle_score,'oracle_gain_vs_fixed':fixed_score-oracle_score,'loo_tree_score':float(c.chosen_score.mean()),'loo_gain_vs_fixed':float(c.fixed_score.mean()-c.chosen_score.mean()),'loo_oracle_method_accuracy':float((c.chosen_method==c.oracle_method).mean()),'mean_pareto_methods':float(s[s.pareto].groupby('policy_key').size().mean())})
    out=pd.DataFrame(summary); out.to_csv(a.output/'summary.csv',index=False); pd.DataFrame(routing).to_csv(a.output/'loo_router.csv',index=False)
    lines=['# Dynamic graph solver-router analysis','','Scores are within-policy percentile ranks; lower is better.','','| Profile | Policies | Best fixed | Fixed score | Oracle score | Oracle gain | LOO tree score | LOO gain | Router accuracy | Pareto methods |','| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in out.itertuples(index=False): lines.append(f'| {r.profile} | {r.policies} | {r.best_fixed_method} | {r.best_fixed_score:.4f} | {r.oracle_router_score:.4f} | {r.oracle_gain_vs_fixed:.4f} | {r.loo_tree_score:.4f} | {r.loo_gain_vs_fixed:.4f} | {r.loo_oracle_method_accuracy:.4f} | {r.mean_pareto_methods:.2f} |')
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n'); (a.output/'run_metadata.json').write_text(json.dumps({'status':'executed','scope':'posthoc one-seed hypothesis screen; no SOTA claim','profiles':list(PROFILES)},indent=2)); print(out.to_string(index=False))
if __name__=='__main__': main()
