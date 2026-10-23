#!/usr/bin/env bash
#SBATCH --job-name=llmtd-varfront
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${VAR_DATASET:?}"; : "${VAR_CATALOG:?}"; : "${VAR_SEED:?}"; : "${VAR_LABEL:?}"; : "${VAR_ADAPTER:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$VAR_DATASET" in
  adult) D="$R/full_train_1028011" ;;
  default) D="$R/full_train_default_1033760" ;;
  shoppers) D="$R/full_train_shoppers_1033761" ;;
  *) exit 2 ;;
esac
OUT="$R/adapter_variant_frontier_${VAR_LABEL}_${VAR_DATASET}_${VAR_CATALOG}_seed${VAR_SEED}_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/rejection_frontier.py" --repo "$REPO" --dataset "$VAR_DATASET" \
  --catalog "$VAR_CATALOG" --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" \
  --adapter "$VAR_ADAPTER" --output "$OUT" --pool-samples 65536 --conditional-samples 2048 \
  --steps 30 --guidance-scales 1:2:4 --run-label "$VAR_LABEL" --seed "$VAR_SEED"
