#!/usr/bin/env bash
# One-epoch integration check for frozen-teacher graph targets and backprop.
#SBATCH --job-name=llmtd-news-e2e-smoke
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
TRAIN="$A/outputs/full_train_news_official_regression_1093974"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
OUT="$A/outputs/news_e2e_teacher_smoke_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/finetune_trajectory_teacher_tabdiff.py" \
  --repo "$REPO" --dataset news --regression-target-mode numerical \
  --config "$TRAIN/config.pkl" --checkpoint "$TRAIN/ema_model_epoch_4000.pt" \
  --graph "$GRAPH" --output "$OUT" --epochs 1 --batch-size 4096 \
  --trainable-scope head --target-mode sparse --teacher-transport direct \
  --teacher-strength .5 --auxiliary-weight .05 --anchor-weight 1 \
  --t-min .10 --t-max .40 --seed 20261025

