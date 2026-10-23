# LLM feedback during tabular diffusion — exploratory run

Latest confirmed three-dataset results, metric policy, negative results and
paper outline (Vietnamese, 2026-10-11):
[`PAPER_RESULTS_OUTLINE_20261011_VI.md`](PAPER_RESULTS_OUTLINE_20261011_VI.md).

Latest consolidated proposal and experiment synthesis (Vietnamese, 2026-10-08):
[`PROPOSAL_DYNAMIC_GRAPH_TONG_HOP_20261008_VI.md`](PROPOSAL_DYNAMIC_GRAPH_TONG_HOP_20261008_VI.md).

Focused results, gains, baseline provenance, venue and publication dates:
[`KET_QUA_BASELINE_VA_PUBLICATION_20261008_VI.md`](KET_QUA_BASELINE_VA_PUBLICATION_20261008_VI.md).

Catch-up summary (Vietnamese, 2026-10-02):
[`CATCHUP_TONG_HOP_20261002_VI.md`](CATCHUP_TONG_HOP_20261002_VI.md).

Frozen comparison of paper-reported baselines, official self-runs, and project
results (Vietnamese, 2026-10-02):
[`BANG_TONG_HOP_PAPER_BASELINE_VA_KET_QUA_20261002.md`](BANG_TONG_HOP_PAPER_BASELINE_VA_KET_QUA_20261002.md).

Dynamic-graph visual audit (dataset graphs and timestep activation heatmaps):
[`outputs/dynamic_graph_visualization_20261002_v3/RESULTS.md`](outputs/dynamic_graph_visualization_20261002_v3/RESULTS.md).

Current full project synthesis and general-benchmark plan (Vietnamese):
[`BAO_CAO_TONG_HOP_GENERAL_DYNAMIC_GRAPH_20261002.md`](BAO_CAO_TONG_HOP_GENERAL_DYNAMIC_GRAPH_20261002.md).

Latest gap/limitation synthesis: [`GAP_LIMITATION_LEDGER_20260928_VI.md`](GAP_LIMITATION_LEDGER_20260928_VI.md).

Newest result tables:

- `outputs/hybrid_projection_aggregate_1044393/RESULTS.md`: compute-matched structured-condition → exact-certification results;
- `outputs/ctdf_style_cert_aggregate_1044401/RESULTS.md`: CTDF-style numeric proxy with raw-space certification;
- `outputs/mixed_condition_aggregate_1044417/RESULTS.md`: three-seed mixed numeric/categorical contract transfer;
- `outputs/mixed_condition_decomp_aggregate_1044463/summary.csv`: target/non-target fidelity decomposition;
- `outputs/mixed_type_claim_audit_1044467/RESULTS.md`: paired contract–seed bootstrap;
- `outputs/harpoon_style_aggregate_1044475/RESULTS.md`: same-backbone Harpoon-style gradient proxy sweep;
- `outputs/request_condition_aggregate_1044443/RESULTS.md`: fresh-schema compiler stress test across Default and Shoppers.

This directory is isolated from existing checkpoints and results.  It contains
mechanism pilots and proxy baselines; nothing here is labelled as a faithful
paper reproduction unless explicitly stated in the final report.

## Slurm execution

Submit from `/home/cs.aau.dk/tungkvt/binhtt2`:

The primary rerun path below uses a unique output directory, so it does not
overwrite the evidence delivered with this analysis. The two API jobs are
optional; they only send schema/validated rules and synthetic preview rows.

```bash
analysis=baseline_repos/tabdiff/analysis/llm_feedback_20260922
run_root="$(pwd)/$analysis/outputs/rerun_$(date +%Y%m%d_%H%M%S)"
export_arg="ALL,ANALYSIS_RUN_ROOT=$run_root"

profile_id=$(sbatch --parsable --export="$export_arg" "$analysis/slurm/01_profile_data.sbatch")
train_id=$(sbatch --parsable --export="$export_arg" "$analysis/slurm/02_train_pilot.sbatch")
pilot_id=$(sbatch --parsable --export="$export_arg" --dependency=afterok:"$train_id" "$analysis/slurm/06_probe_pilot_late_intervention.sbatch")
verify_id=$(sbatch --parsable --export="$export_arg" --dependency=afterok:"$pilot_id" "$analysis/slurm/07_independent_verifier.sbatch")
llm_id=$(sbatch --parsable --export="$export_arg" --dependency=afterok:"$pilot_id" "$analysis/slurm/08_llm_audit_retry.sbatch")
llm_rules_id=$(sbatch --parsable --export="$export_arg" --dependency=afterok:"$pilot_id" "$analysis/slurm/09_llm_audit_static_rules.sbatch")
```

One-shot status check (continuous tracking is unnecessary):

```bash
sacct -j "$profile_id,$train_id,$pilot_id,$verify_id,$llm_id,$llm_rules_id" \
  --format=JobID,JobName,State,Elapsed,ExitCode,AllocTRES,MaxRSS,NodeList
```

## Where to inspect results

For the full current synthesis in Vietnamese, start with
`PROJECT_SYNTHESIS_20260926_VI.md`. It separates executed, invalidated, pending,
and inferred evidence; it also records the related-work boundary, all formal
hypotheses, the newest cross-dataset checkpoints, the two-phase gate result,
and the target-scoped actuation jobs. `DYNAMIC_SEMANTIC_CONSTRAINTS_20260925.md`
preserves the immediately preceding narrowed formulation. `outputs/job_manifest.json`
is the provenance/status index for every submitted job.

- `outputs/profile/`: dataset/column profiles, train-only rule candidates, and
  held-out rule violations for Adult, Default, and Shoppers.
- `outputs/debug_probe/`: execution/sanity evidence from the existing four-epoch
  debug checkpoint.  Do not use it for a model-quality claim.
- `outputs/pilot_train/`: 500-epoch bounded pilot checkpoint and learning curve.
- `outputs/pilot_probe/`: known-x0 and free-generation trajectory tables,
  trajectory plot, samples, and intervention metrics.
- `outputs/llm_audit_v2/` and `outputs/llm_audit_static_rules/`: per-call
  structured critiques, verifier comparison, repeated-call disagreement,
  tokens, and runtime. Only schema metadata, aggregate validated relations,
  and synthetic preview records are sent to the API.
- `REPORT.md`: final evidence table, related-work positioning, hypotheses, and
  go/no-go decision.
- `NICHE_FOLLOWUP.md`: follow-up on temporal persistence, selective recheck,
  critique contracts, typed actuators, and the full-checkpoint validation gate.

The quickest visual check is `outputs/pilot_probe_v2/trajectory_readability.png`.
The main quantitative tables are `outputs/pilot_probe_v2/intervention_metrics.csv`,
`outputs/pilot_probe_v2/independent_verifier_metrics.csv`, and the LLM audit
summary JSON files.

## Submitted run (2026-09-22)

| Job | ID | Dependency |
|---|---:|---|
| dataset profile/rule audit | `1027375` | none |
| bounded TabDiff pilot train | `1027373` | none |
| existing debug-checkpoint probe (failed: missing `__len__`, preserved for provenance) | `1027374` | none |
| existing debug-checkpoint probe retry | `1027379` | none |
| pilot-checkpoint trajectory/interventions | `1027376` | after successful `1027373` |
| LLM feedback audit | `1027377` | after successful `1027376` |
| pilot retry with evidence-driven late `t=0.10` intervention (separate output) | `1027386` | none; reuses completed pilot checkpoint |
| withheld-rule and exception-subgroup audit | `1027387` | after successful `1027386` |
| LLM audit retry with callable `gpt-4.1-mini` (separate output) | `1027389` | none; original attempt was blocked by organization verification for `gpt-5-mini` |
| LLM audit with static validated relations (failed transient preflight) | `1027392` | preserved for provenance |
| LLM audit with static validated relations retry (cancelled after rate limit) | `1027393` | preserved for provenance |
| Rate-limited static-relation retry | `1027394` | completed; 18/18 calls, 7 s pacing, 3:18 wall time |
