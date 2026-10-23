#!/usr/bin/env python3
"""Does mid-denoising intervention add value beyond strong non-mid baselines?"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS=['active_drift_sd','active_target_ks_reference','active_target_wasserstein_sd','active_cohort_numeric_ks','active_cohort_categorical_tv','target_context_spearman_mae']
PROFILES={'balanced':np.ones(6),'intervention_conservative':np.array([3,1,1,1,1,1]),'target_fidelity':np.array([1,3,3,1,1,1]),'joint_fidelity':np.array([1,1,1,3,3,3])}
GROUPS={
 'mid_denoising':['lookahead_anchored_violated_masked_t100_plus_certifier','lookahead_anchored_violated_topk3_t100_plus_certifier','lookahead_anchored_violated_topk5_t100_plus_certifier','lookahead_anchored_violated_numgraph_t100_plus_certifier','lookahead_anchored_violated_t100_plus_certifier'],
 'final_repair':['subgroup_final_repair','empirical_marginal_bad_postrepair','empirical_stratified_bad_postrepair'],
 'conditional_resampling':['diffusion_impute_then_certify','diffusion_impute_numgraph_then_certify','diffusion_bestof8_target_then_certify','diffusion_bestof8_numgraph_then_certify'],
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--default',type=Path,required=True);p.add_argument('--shoppers',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 frames=[]
 for ds,path in [('default',a.default),('shoppers',a.shoppers)]:
  x=pd.read_csv(path/'policy_metrics.csv');x['dataset']=ds;frames.append(x)
 d=pd.concat(frames,ignore_index=True);d['policy_key']=d.dataset+':'+d.policy;reverse={m:g for g,ms in GROUPS.items() for m in ms};d=d[d.method.isin(reverse)&d.active_satisfaction.ge(.999)&d.exception_changed.fillna(0).le(1e-9)].copy();d['stage_group']=d.method.map(reverse)
 rows=[]
 for key,g in d.groupby('policy_key'):
  ranks=g[METRICS].rank(method='average',pct=True).fillna(1.)
  for profile,w in PROFILES.items():
   score=(ranks.to_numpy()*w).sum(1)/w.sum()
   for (_,r),s in zip(g.iterrows(),score):rows.append({'policy_key':key,'dataset':r.dataset,'policy':r.policy,'profile':profile,'stage_group':r.stage_group,'method':r.method,'score':float(s),**{m:r[m] for m in METRICS}})
 scored=pd.DataFrame(rows);scored.to_csv(a.output/'method_scores.csv',index=False)
 best=scored.loc[scored.groupby(['policy_key','profile','stage_group']).score.idxmin()].copy();best.to_csv(a.output/'best_by_stage.csv',index=False)
 summary=[];pairs=[]
 for profile,pdta in best.groupby('profile'):
  pivot=pdta.pivot(index='policy_key',columns='stage_group',values='score').dropna()
  for key,r in pivot.iterrows():
   nonmid=min(r.final_repair,r.conditional_resampling);pairs.append({'profile':profile,'policy_key':key,'mid':r.mid_denoising,'final':r.final_repair,'resampling':r.conditional_resampling,'best_nonmid':nonmid,'mid_gain_vs_best_nonmid':nonmid-r.mid_denoising})
  pr=pd.DataFrame([x for x in pairs if x['profile']==profile]);winner=pdta.loc[pdta.groupby('policy_key').score.idxmin()].stage_group.value_counts()
  summary.append({'profile':profile,'policies':len(pr),'mid_score':pr.mid.mean(),'final_score':pr.final.mean(),'resampling_score':pr.resampling.mean(),'best_nonmid_score':pr.best_nonmid.mean(),'mid_gain_vs_best_nonmid':pr.mid_gain_vs_best_nonmid.mean(),'mid_win_fraction':(pr.mid_gain_vs_best_nonmid>0).mean(),'mid_tie_fraction':np.isclose(pr.mid_gain_vs_best_nonmid,0).mean(),'global_wins_mid':int(winner.get('mid_denoising',0)),'global_wins_final':int(winner.get('final_repair',0)),'global_wins_resampling':int(winner.get('conditional_resampling',0))})
 out=pd.DataFrame(summary);pd.DataFrame(pairs).to_csv(a.output/'paired_policy_scores.csv',index=False);out.to_csv(a.output/'summary.csv',index=False)
 lines=['# Incremental value of mid-denoising intervention','','All compared outputs satisfy the active constraint and preserve exceptions. Scores are within-policy ranks; lower is better.','','| Profile | Policies | Mid | Final | Resampling | Best non-mid | Mid gain | Mid win fraction | Wins mid/final/resample |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
 for r in out.itertuples(index=False):lines.append(f'| {r.profile} | {r.policies} | {r.mid_score:.4f} | {r.final_score:.4f} | {r.resampling_score:.4f} | {r.best_nonmid_score:.4f} | {r.mid_gain_vs_best_nonmid:.4f} | {r.mid_win_fraction:.4f} | {r.global_wins_mid}/{r.global_wins_final}/{r.global_wins_resampling} |')
 (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');(a.output/'run_metadata.json').write_text(json.dumps({'status':'executed','scope':'same one-seed policy grid','validity_filter':'active satisfaction >= .999 and no exception change','interpretation':'hypothesis screen, not SOTA'},indent=2));print(out.to_string(index=False))
if __name__=='__main__':main()
