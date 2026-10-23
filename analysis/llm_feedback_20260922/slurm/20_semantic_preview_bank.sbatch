#!/usr/bin/env bash
#SBATCH --job-name=llmtd-sembank
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/semantic_preview_bank.py" \
  --repo "$REPO" \
  --checkpoint "$ANALYSIS/outputs/full_train_1028011/ema_model_epoch_8000.pt" \
  --config "$ANALYSIS/outputs/full_train_1028011/config.pkl" \
  --output "$RUN_ROOT/semantic_bank_${SLURM_JOB_ID}" \
  --samples 96 \
  --steps 30 \
  --seed 20260931 \
  --cases-per-type 4 \
  --device cuda
