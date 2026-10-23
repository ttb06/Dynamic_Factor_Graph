#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xact
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${XACT_DATASET:?}"; : "${XACT_SEED:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"; R="${ANALYSIS_RUN_ROOT:-$A/outputs}"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$XACT_DATASET" in default) D="$R/full_train_default_1033760";; shoppers) D="$R/full_train_shoppers_1033761";; *) exit 2;; esac
cd "$REPO"; "$PY" "$A/scripts/cross_dataset_actuation_baselines.py" --repo "$REPO" --dataset "$XACT_DATASET" --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" --output "$R/${XACT_TAG:-cross_actuation}_${XACT_DATASET}_seed${XACT_SEED}_${SLURM_JOB_ID}" --samples 512 --steps 30 --seed "$XACT_SEED"
