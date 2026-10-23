#!/usr/bin/env bash
#SBATCH --job-name=llmtd-paired-portfolio
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers default)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}

if [[ "$dataset" == shoppers ]]; then
  graph="$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"
else
  graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
fi

"$PY" "$A/scripts/paired_trajectory_portfolio.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$graph" --evaluation-graph "$evaluation" \
  --input-glob "$A/outputs/feedback_uncapped_${dataset}_*_${FEEDBACK_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  --edit-weights 0.1,0.5,1.0,2.0 \
  --output "$A/outputs/paired_portfolio_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
