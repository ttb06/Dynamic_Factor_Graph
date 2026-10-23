#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xckpt
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${CROSS_DATASET:?missing dataset}"
: "${CROSS_TRAIN_JOB:?missing training job id}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
TRAIN_DIR="$RUN_ROOT/full_train_${CROSS_DATASET}_${CROSS_TRAIN_JOB}"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/cross_dataset_checkpoint_audit.py" \
  --repo "$REPO" --dataset "$CROSS_DATASET" \
  --config "$TRAIN_DIR/config.pkl" --checkpoint "$TRAIN_DIR/ema_model_epoch_8000.pt" \
  --output "$RUN_ROOT/cross_dataset_checkpoint_${CROSS_DATASET}_${SLURM_JOB_ID}" \
  --samples "${CROSS_AUDIT_SAMPLES:-4096}" --steps "${CROSS_AUDIT_STEPS:-30}" \
  --seed "${CROSS_AUDIT_SEED:-20260925}"
