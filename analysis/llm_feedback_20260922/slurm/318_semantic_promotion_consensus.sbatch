#!/usr/bin/env bash
#SBATCH --job-name=llmtd-promote-consensus
#SBATCH --partition=cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers default diabetes)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
if [[ "$dataset" == shoppers ]]; then graph="$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json"
elif [[ "$dataset" == default ]]; then graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"
else graph="$A/outputs/trajectory_portfolio_v2_diabetes_1060979_2/constraint_catalog.json"; fi
"$PY" "$A/scripts/semantic_promotion_consensus.py" --graph "$graph" \
  --review-a "$A/outputs/semantic_promotion_${dataset}_${REVIEW_JOB_A}_${SLURM_ARRAY_TASK_ID}" \
  --review-b "$A/outputs/semantic_promotion_${dataset}_${REVIEW_JOB_B}_${SLURM_ARRAY_TASK_ID}" \
  --output "$A/outputs/semantic_promotion_consensus_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
