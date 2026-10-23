# Code walkthrough

The analysis directory contains chronological research scripts. Start with `research_structure/README.md`, whose Construct → Guide → Repair façade identifies the conceptual core without moving or deleting provenance-linked history. The concrete entrypoints below remain at their original paths.

## Dynamic generation

- `analysis/llm_feedback_20260922/scripts/dense_numeric_factor_messages_arr.py`  
  Main array-friendly driver used by later multi-dataset runs. It loads a TabDiff config/checkpoint, constructs the requested factor actions, generates paired unconditional/guided pools, and writes run metadata.

- `analysis/llm_feedback_20260922/scripts/dense_numeric_factor_messages.py`  
  Earlier/main driver variant used by several categorical and numerical factor experiments.

- `analysis/llm_feedback_20260922/scripts/categorical_logit_guidance.py`  
  Categorical marginal/target/pair messages applied before masked categorical updates.

## Generic C3 / particle selection

- `analysis/llm_feedback_20260922/scripts/balanced_particle_resampling.py`  
  Generic train-only marginal/pair raking, consensus priors, geometric coverage, and two-fold integrated coverage. This is the frozen `old` selector family.

## Graph-aware selection

- `analysis/llm_feedback_20260922/scripts/graph_consistent_particle_resampling.py`  
  Factor loading, state encoding, relation/coverage/hard losses, and scalar graph-aware selection.

- `analysis/llm_feedback_20260922/scripts/graph_pareto_anchored_resampling.py`  
  Frozen-anchor graph feasibility constraints with full-subset redraw.

- `analysis/llm_feedback_20260922/scripts/graph_proximal_swap_repair.py`  
  Current proposed set-level method. It computes factor-directed add/remove scores, samples sparse membership swaps, enforces train-only constraints, uses quality-first selection, and supports deficit-activated no-op behavior.

## Audit and evaluation

- `analysis/llm_feedback_20260922/scripts/audit_graph_consistency.py`  
  Shared external audit for graph relation loss, factor coverage gap, and hard excess.

- `analysis/llm_feedback_20260922/scripts/evaluate_shoppers_baseline_suite.py`  
  Common multi-dataset evaluator used for Shape, Trend, Alpha, Beta, utility, privacy, and graph diagnostics.

- `analysis/llm_feedback_20260922/scripts/summarize_graph_pareto_screen.py`  
  Fail-closed screen aggregation.

- `analysis/llm_feedback_20260922/scripts/summarize_graph_pareto_confirmation.py`  
  Fresh/held-out confirmation aggregation and decisions.

## Current launchers

- `slurm/899_adult_graph_proximal_transfer_sweep.sbatch`
- `slurm/900_default_graph_proximal_transfer_sweep.sbatch`
- `slurm/902_graph_deficit_gate_smoke.sbatch`
- `slurm/903_confirm_graph_deficit_gate.sbatch`
- `slurm/904_summarize_graph_deficit_gate_confirmation.sbatch`

Historical numbered launchers are intentionally retained because they encode negative-result protocols and exact provenance.

## Reading a result directory

Typical structure:

```text
output_root/
  provenance.txt
  pool/run_metadata.json
  graph/selection_metadata.json
  graph/selected_indices.csv
  graph_audit/graph_metrics.csv
  eval/baseline_metrics.csv
  eval/RESULTS.md
```

The Git package excludes heavy pool/sample CSVs but preserves the metadata, selected-index files when small, summaries, and manifests.

