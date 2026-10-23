#!/usr/bin/env bash
#SBATCH --job-name=llmtd-match-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${GLOBAL_EVAL_JOB_ID:?GLOBAL_EVAL_JOB_ID required}"
: "${FINAL_EVAL_JOB_ID:?FINAL_EVAL_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/aggregate_matched_mechanism_ablation.py" \
  --root "$A/outputs" --mid-eval-job 1071210 \
  --global-eval-job "$GLOBAL_EVAL_JOB_ID" --final-eval-job "$FINAL_EVAL_JOB_ID" \
  --output "$A/outputs/matched_mechanism_aggregate_${SLURM_JOB_ID}"
