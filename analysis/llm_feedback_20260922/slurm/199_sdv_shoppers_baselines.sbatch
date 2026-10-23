#!/usr/bin/env bash
#SBATCH --job-name=llmtd-sdv-base
#SBATCH --partition=prioritized
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
METHODS=(ctgan tvae); METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/train_sdv_shoppers_baseline.py" --repo "$REPO" --method "$METHOD" \
  --epochs 300 --samples 4096 --seed 20261007 \
  --output "$A/outputs/sdv_shoppers_${METHOD}_${SLURM_ARRAY_JOB_ID}"
