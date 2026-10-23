#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xactive
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${ACTIVE_DATASET:?}"; : "${ACTIVE_SEED:?}"; : "${ACTIVE_GUIDANCE:?}"; : "${ACTIVE_ADAPTER:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"; R="${ANALYSIS_RUN_ROOT:-$A/outputs}"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$ACTIVE_DATASET" in default) D="$R/full_train_default_1033760";; shoppers) D="$R/full_train_shoppers_1033761";; *) exit 2;; esac
OUT="$R/cross_active_${ACTIVE_DATASET}_g${ACTIVE_GUIDANCE}_seed${ACTIVE_SEED}_${SLURM_JOB_ID}"; cd "$REPO"
"$PY" "$A/scripts/cross_dataset_active_set.py" --repo "$REPO" --dataset "$ACTIVE_DATASET" --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" --adapter "$ACTIVE_ADAPTER" --output "$OUT" --samples 2048 --steps 30 --guidance "$ACTIVE_GUIDANCE" --seed "$ACTIVE_SEED"
