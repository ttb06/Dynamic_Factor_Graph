#!/usr/bin/env bash
#SBATCH --job-name=llmtd-union-eval
#SBATCH --partition=cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${CATALOG_JOB_ID:?CATALOG_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers default diabetes)
OLDS=(expanded_constraint_catalog_1057680 cross_dataset_catalog_default_1058707_1 diabetes_catalog_1060368)
i=$SLURM_ARRAY_TASK_ID; dataset=${DATASETS[$i]}; old=${OLDS[$i]}
"$PY" "$A/scripts/build_union_evaluation_graph.py" \
  --old "$A/outputs/$old/constraint_catalog.json" \
  --new "$A/outputs/actionable_catalog_v2_${dataset}_${CATALOG_JOB_ID}_${i}/constraint_catalog.json" \
  --output "$A/outputs/actionable_union_eval_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
