#!/usr/bin/env bash
# Train-only edge/time benefit audit for automatic dynamic-graph scheduling.
#SBATCH --job-name=llmtd-news-edge-benefit
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
TRAIN="$A/outputs/full_train_news_official_regression_1093974"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
OUT="$A/outputs/news_edge_benefit_time_${SLURM_JOB_ID}"

cd "$REPO"
"$PY" "$A/scripts/audit_edge_action_benefit_by_time.py" \
  --repo "$REPO" --dataset news --regression-target-mode numerical \
  --config "$TRAIN/config.pkl" --checkpoint "$TRAIN/ema_model_epoch_4000.pt" \
  --graph "$GRAPH" --rows 8192 --batch-size 2048 \
  --times .95,.75,.55,.40,.30,.20,.12,.07,.03 --strength .50 \
  --seed 20261031 --output "$OUT"
