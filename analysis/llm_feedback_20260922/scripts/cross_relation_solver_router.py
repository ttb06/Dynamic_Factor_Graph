#!/usr/bin/env python3
"""Leave-one-relation-family-out solver performance router."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

METRICS=['active_drift_sd','active_target_ks_reference','active_target_wasserstein_sd','active_cohort_numeric_ks','active_cohort_categorical_tv','target_context_spearman_mae']
PROFILES={'balanced':np.ones(6),'intervention_conservative':np.array([3,1,1,1,1,1]),'target_fidelity':np.array([1,3,3,1,1,1]),'joint_fidelity':np.array([1,1,1,3,3,3])}
KEEP=['subgroup_final_repair','empirical_marginal_bad_postrepair','empirical_stratified_bad_postrepair','diffusion_impute_then_certify','diffusion_impute_numgraph_then_certify','diffusion_bestof8_target_then_certify','diffusion_bestof8_numgraph_then_certify','lookahead_anchored_violated_masked_t100_plus_certifier','lookahead_anchored_violated_topk3_t100_plus_certifier','lookahead_anchored_violated_topk5_t100_plus_certifier','lookahead_anchored_violated_numgraph_t100_plus_certifier','lookahead_anchored_violated_t100_plus_certifier']

def family(policy): return re.sub(r'_[-+]?\d+(?:\.\d+)?$','',str(policy))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--default',type=Path,required=True); p.add_argument('--shoppers',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    parts=[]
    for ds,path in [('default',a.default),('shoppers',a.shoppers)]:
        x=pd.read_csv(path/'policy_metrics.csv'); x['dataset']=ds; parts.append(x)
    raw=pd.concat(parts,ignore_index=True); raw['policy_key']=raw.dataset+':'+raw.policy; raw['family']=raw.dataset+':'+raw.policy.map(family)
    unc=raw[raw.method.eq('unconditional')].set_index('policy_key')
    valid=raw[raw.method.isin(KEEP)&raw.active_satisfaction.ge(.999)&raw.exception_changed.fillna(0).le(1e-9)].copy()
    feature_rows=[]
    for key,g in valid.groupby('policy_key'):
        u=unc.loc[key]
        for _,r in g.iterrows():
            feature_rows.append({'policy_key':key,'family':r.family,'dataset':r.dataset,'regime':r.regime,'method':r.method,'threshold':r.threshold,'difficulty':1-u.active_satisfaction,'active_n':u.active_n,'exception_n':u.exception_n,'uncond_drift':u.active_drift_sd,'uncond_target_ks':u.active_target_ks_reference,'uncond_target_w':u.active_target_wasserstein_sd,'uncond_num_ks':u.active_cohort_numeric_ks,'uncond_cat_tv':u.active_cohort_categorical_tv,'uncond_corr_mae':u.target_context_spearman_mae,**{m:r[m] for m in METRICS}})
    data=pd.DataFrame(feature_rows)
    score_rows=[]
    for key,g in data.groupby('policy_key'):
        ranks=g[METRICS].rank(method='average',pct=True).fillna(1.)
        for profile,w in PROFILES.items():
            scores=(ranks.to_numpy()*w).sum(1)/w.sum()
            for (idx,r),s in zip(g.iterrows(),scores): score_rows.append({**r.to_dict(),'profile':profile,'score':float(s)})
    scores=pd.DataFrame(score_rows); scores.to_csv(a.output/'training_rows.csv',index=False)
    num=['threshold','difficulty','active_n','exception_n','uncond_drift','uncond_target_ks','uncond_target_w','uncond_num_ks','uncond_cat_tv','uncond_corr_mae']; cat=['dataset','regime','method']
    choices=[]
    for profile,pdata in scores.groupby('profile'):
        for held in sorted(pdata.family.unique()):
            train=pdata[pdata.family.ne(held)]; test=pdata[pdata.family.eq(held)]
            fixed=train.groupby('method').score.mean().idxmin()
            model=Pipeline([('prep',ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])),('rf',RandomForestRegressor(n_estimators=160,max_depth=7,min_samples_leaf=2,max_features=.8,random_state=20261001,n_jobs=4))])
            model.fit(train[num+cat],train.score); test=test.copy(); test['pred']=model.predict(test[num+cat])
            for key,g in test.groupby('policy_key'):
                chosen=g.loc[g.pred.idxmin()]; oracle=g.loc[g.score.idxmin()]; fixedrow=g[g.method.eq(fixed)]
                if fixedrow.empty: fixedrow=g.loc[[g.score.idxmin()]]
                fixedrow=fixedrow.iloc[0]
                choices.append({'profile':profile,'held_family':held,'policy_key':key,'chosen_method':chosen.method,'chosen_score':chosen.score,'oracle_method':oracle.method,'oracle_score':oracle.score,'fixed_method':fixedrow.method,'fixed_score':fixedrow.score})
    c=pd.DataFrame(choices); c.to_csv(a.output/'heldout_choices.csv',index=False)
    out=c.groupby('profile').agg(policies=('policy_key','nunique'),families=('held_family','nunique'),router_score=('chosen_score','mean'),fixed_score=('fixed_score','mean'),oracle_score=('oracle_score','mean'),router_accuracy=('chosen_method',lambda x:float((x==c.loc[x.index,'oracle_method']).mean()))).reset_index(); out['gain_vs_fixed']=out.fixed_score-out.router_score; out['remaining_oracle_gap']=out.router_score-out.oracle_score; out.to_csv(a.output/'summary.csv',index=False)
    lines=['# Cross-relation solver routing','','Each relation family, including all of its thresholds, is held out together. Lower normalized rank is better.','','| Profile | Policies | Families | Router | Fold-fixed | Oracle | Gain vs fixed | Oracle gap | Exact choice |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in out.itertuples(index=False):lines.append(f'| {r.profile} | {r.policies} | {r.families} | {r.router_score:.4f} | {r.fixed_score:.4f} | {r.oracle_score:.4f} | {r.gain_vs_fixed:.4f} | {r.remaining_oracle_gap:.4f} | {r.router_accuracy:.4f} |')
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n'); (a.output/'run_metadata.json').write_text(json.dumps({'status':'executed','validation':'leave-one-relation-family-out','scope':'one-seed hypothesis screen','no_sota_claim':True},indent=2)); print(out.to_string(index=False))
if __name__=='__main__':main()
