#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xcond
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${XCOND_DATASET:?}"; : "${XCOND_REP:?}"; : "${XCOND_SEED:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"; R="${ANALYSIS_RUN_ROOT:-$A/outputs}"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$XCOND_DATASET" in default) D="$R/full_train_default_1033760";; shoppers) D="$R/full_train_shoppers_1033761";; *) exit 2;; esac
OUT="$R/cross_condition_${XCOND_DATASET}_${XCOND_REP}_seed${XCOND_SEED}_${SLURM_JOB_ID}"; cd "$REPO"
"$PY" "$A/scripts/cross_dataset_condition_adapter.py" --repo "$REPO" --dataset "$XCOND_DATASET" --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" --output "$OUT" --representation "$XCOND_REP" --epochs 100 --pairs-per-contract 2048 --eval-samples 512 --guidance-scales 1:2:4 --seed "$XCOND_SEED"
