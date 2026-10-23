#!/usr/bin/env bash
#SBATCH --job-name=llmtd-hybridpx
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${HYBRID_DATASET:?}"; : "${HYBRID_SEED:?}"; : "${HYBRID_ADAPTER:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"; R="${ANALYSIS_RUN_ROOT:-$A/outputs}"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$HYBRID_DATASET" in default) D="$R/full_train_default_1033760";; shoppers) D="$R/full_train_shoppers_1033761";; *) exit 2;; esac
OUT="$R/hybrid_projection_${HYBRID_DATASET}_seed${HYBRID_SEED}_${SLURM_JOB_ID}"; cd "$REPO"
"$PY" "$A/scripts/cross_dataset_condition_large_eval.py" --repo "$REPO" --dataset "$HYBRID_DATASET" --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" --adapter "$HYBRID_ADAPTER" --output "$OUT" --samples 4096 --guidance-scales "${HYBRID_GUIDANCE:-2:4}" --include-final-projection --seed "$HYBRID_SEED"
