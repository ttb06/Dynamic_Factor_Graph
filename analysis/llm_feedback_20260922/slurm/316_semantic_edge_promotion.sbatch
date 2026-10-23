#!/usr/bin/env bash
#SBATCH --job-name=llmtd-semantic-promote
#SBATCH --partition=cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
ENV_FILE=/home/cs.aau.dk/tungkvt/binhtt2/EIPC/.env
DATASETS=(shoppers default diabetes)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
if [[ "$dataset" == shoppers ]]; then
  graph="$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json"
elif [[ "$dataset" == default ]]; then
  graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"
else
  graph="$A/outputs/trajectory_portfolio_v2_diabetes_1060979_2/constraint_catalog.json"
fi
"$PY" "$A/scripts/semantic_edge_promotion.py" --dataset "$dataset" --graph "$graph" \
  --env-file "$ENV_FILE" --model "${SEMANTIC_MODEL:-gpt-5.4}" --min-confidence .80 \
  --output "$A/outputs/semantic_promotion_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
