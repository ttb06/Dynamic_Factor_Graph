#!/usr/bin/env bash
#SBATCH --job-name=llmtd-pilotprobe
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/tabdiff_probe.py" \
  --repo "$REPO" \
  --checkpoint "$ANALYSIS/outputs/pilot_train/ema_model_epoch_500.pt" \
  --config "$ANALYSIS/outputs/pilot_train/config.pkl" \
  --output "$ANALYSIS/outputs/pilot_probe" \
  --samples 768 \
  --probe-rows 384 \
  --steps 30 \
  --seed 20260922 \
  --device cuda
