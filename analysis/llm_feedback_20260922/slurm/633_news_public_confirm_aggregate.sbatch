#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CONFIRM_ARRAY_ID:?CONFIRM_ARRAY_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/aggregate_news_public_confirm.py" \
  --evaluation "$A/outputs/news_public_progressive_eval_1081003" \
  --evaluation "$A/outputs/news_public_progressive_eval_${CONFIRM_ARRAY_ID}_0" \
  --evaluation "$A/outputs/news_public_progressive_eval_${CONFIRM_ARRAY_ID}_1" \
  --output "$A/outputs/news_public_confirm_aggregate_${SLURM_JOB_ID}"
