#!/usr/bin/env bash
#SBATCH --job-name=llmtd-persist
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/persistence_feedback_probe.py" \
  --repo "$REPO" \
  --checkpoint "$ANALYSIS/outputs/pilot_train/ema_model_epoch_500.pt" \
  --config "$ANALYSIS/outputs/pilot_train/config.pkl" \
  --output "$RUN_ROOT/niche_persistence_${SLURM_JOB_ID}" \
  --samples 512 \
  --steps 30 \
  --seeds 20260922 20260923 20260924 \
  --intervention-t 0.10 \
  --recheck-t 0.03 \
  --device cuda
