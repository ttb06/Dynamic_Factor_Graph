#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-project-a
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${EVALUATION_JOB_ID:?EVALUATION_JOB_ID required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
"$PY" "$A/scripts/aggregate_news_projection_ablation.py" \
  --outputs "$A/outputs" --evaluation-job "$EVALUATION_JOB_ID" \
  --control-evaluation "$A/outputs/news_public_progressive_eval_1081003" \
  --output "$A/outputs/news_projection_aggregate_${SLURM_JOB_ID}"
