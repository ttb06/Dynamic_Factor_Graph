#!/usr/bin/env bash
#SBATCH --job-name=llmtd-recent-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${FOREST_EVAL_JOB_ID:?FOREST_EVAL_JOB_ID required}"
: "${TABSYN_EVAL_JOB_ID:?TABSYN_EVAL_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
cd "$REPO"
python3 "$A/scripts/aggregate_recent_baseline_campaign.py" --analysis "$A" \
  --forest-eval-job "$FOREST_EVAL_JOB_ID" --tabsyn-eval-job "$TABSYN_EVAL_JOB_ID" \
  --output "$A/outputs/recent_baseline_campaign_${SLURM_JOB_ID}"
