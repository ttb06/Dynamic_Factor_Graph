#!/usr/bin/env bash
#SBATCH --job-name=llmtd-route-actuation
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SEED=${SEED:-20260930}
ROUNDS=${ROUNDS:-3}
OUT="$A/outputs/ambiguous_diffusion_route_${SLURM_JOB_ID}_seed${SEED}_r${ROUNDS}"
cd "$REPO"
"$PY" "$A/scripts/ambiguous_diffusion_route.py" --repo "$REPO" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --output "$OUT" --steps 30 --rounds "$ROUNDS" --seed "$SEED"
