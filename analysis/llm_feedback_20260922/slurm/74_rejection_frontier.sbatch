#!/usr/bin/env bash
#SBATCH --job-name=llmtd-reject
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${REJECT_DATASET:?}"
: "${REJECT_CATALOG:?}"
: "${REJECT_SEED:?}"
: "${REJECT_ADAPTER:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$REJECT_DATASET" in
  default) D="$R/full_train_default_1033760" ;;
  shoppers) D="$R/full_train_shoppers_1033761" ;;
  *) exit 2 ;;
esac
OUT="$R/rejection_frontier_${REJECT_DATASET}_${REJECT_CATALOG}_seed${REJECT_SEED}_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/rejection_frontier.py" --repo "$REPO" --dataset "$REJECT_DATASET" --catalog "$REJECT_CATALOG" --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" --adapter "$REJECT_ADAPTER" --output "$OUT" --pool-samples 65536 --conditional-samples 4096 --steps 30 --guidance-scales 1:2:4 --seed "$REJECT_SEED"
