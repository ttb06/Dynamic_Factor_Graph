#!/usr/bin/env bash
#SBATCH --job-name=llmtd-fullact
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${FULL_TRAIN_JOB_ID:?submit with --export=ALL,FULL_TRAIN_JOB_ID=<job-id>}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
TRAIN_ROOT="$RUN_ROOT/full_train_${FULL_TRAIN_JOB_ID}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/actuator_probe.py" \
  --repo "$REPO" \
  --checkpoint "$TRAIN_ROOT/ema_model_epoch_8000.pt" \
  --config "$TRAIN_ROOT/config.pkl" \
  --output "$RUN_ROOT/full_actuator_${SLURM_JOB_ID}" \
  --samples 512 \
  --steps 30 \
  --seeds 20260922 20260923 20260924 \
  --device cuda
