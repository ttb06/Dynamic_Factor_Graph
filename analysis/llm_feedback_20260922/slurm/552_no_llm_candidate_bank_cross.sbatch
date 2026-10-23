#!/usr/bin/env bash
#SBATCH --job-name=llmtd-data-bank
#SBATCH --partition=cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(default shoppers diabetes)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
"$PY" "$A/scripts/mine_empirical_bank_no_llm.py" --repo "$REPO" --dataset "$dataset" \
  --maximum 120 --min-support 30 --seed 20261430 \
  --output "$A/outputs/data_candidate_bank_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
