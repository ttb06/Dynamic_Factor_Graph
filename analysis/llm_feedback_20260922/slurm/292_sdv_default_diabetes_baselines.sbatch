#!/usr/bin/env bash
#SBATCH --job-name=llmtd-sdv-cross
#SBATCH --partition=prioritized
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
DATASETS=(default default diabetes diabetes)
METHODS=(ctgan tvae ctgan tvae)
i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}
method=${METHODS[$i]}
samples=8192
if [[ "$dataset" == diabetes ]]; then samples=4096; fi

"$PY" "$A/scripts/train_sdv_shoppers_baseline.py" --repo "$REPO" --dataset "$dataset" \
  --method "$method" --epochs 300 --samples "$samples" --seed 20261096 \
  --output "$A/outputs/sdv_${dataset}_${method}_${SLURM_ARRAY_JOB_ID}_${i}"
