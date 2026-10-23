#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xtrain
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${CROSS_DATASET:?missing default or shoppers}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/train_pilot.py" \
  --repo "$REPO" \
  --base-config "$REPO/debug/ckpt/config.pkl" \
  --dataset "$CROSS_DATASET" \
  --output "$RUN_ROOT/full_train_${CROSS_DATASET}_${SLURM_JOB_ID}" \
  --epochs "${CROSS_EPOCHS:-8000}" \
  --batch-size "${CROSS_BATCH:-4096}" \
  --save-every "${CROSS_SAVE_EVERY:-2000}" \
  --seed "${CROSS_SEED:-20260925}"
