#!/usr/bin/env bash
#SBATCH --job-name=llmtd-ctdfpx
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CTDF_DATASET:?}"; : "${CTDF_SEED:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"; R="${ANALYSIS_RUN_ROOT:-$A/outputs}"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$CTDF_DATASET" in default) D="$R/full_train_default_1033760";; shoppers) D="$R/full_train_shoppers_1033761";; *) exit 2;; esac
OUT="$R/${CTDF_TAG:-ctdf_style}_${CTDF_DATASET}_seed${CTDF_SEED}_${SLURM_JOB_ID}"; cd "$REPO"
"$PY" "$A/scripts/ctdf_style_cross_dataset.py" --repo "$REPO" --dataset "$CTDF_DATASET" --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" --output "$OUT" --samples 1024 --steps 30 --seed "$CTDF_SEED"
