#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixcond
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=05:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${MIXED_DATASET:?}"; : "${MIXED_REP:?}"; : "${MIXED_SEED:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"; R="${ANALYSIS_RUN_ROOT:-$A/outputs}"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$MIXED_DATASET" in default) D="$R/full_train_default_1033760";; shoppers) D="$R/full_train_shoppers_1033761";; *) exit 2;; esac
OUT="$R/${MIXED_TAG:-mixed_condition}_${MIXED_DATASET}_${MIXED_REP}_seed${MIXED_SEED}_${SLURM_JOB_ID}"; cd "$REPO"
"$PY" "$A/scripts/mixed_type_condition_adapter.py" --repo "$REPO" --dataset "$MIXED_DATASET" --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" --output "$OUT" --representation "$MIXED_REP" --epochs 100 --pairs-per-contract 1536 --eval-samples 1024 --guidance-scales 1:2:4 --seed "$MIXED_SEED"
