#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cond-aggregate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${EVAL_JOB_ID:?EVAL_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/aggregate_persistent_trajectory_confirm.py" \
  --root "$A/outputs" --eval-job "$EVAL_JOB_ID" \
  --pattern 'conditional_density_confirm_eval_*_{eval_job}_*/baseline_metrics.csv' \
  --candidate conditional_density_plus_repair \
  --title 'Conditional-density action selection confirmation' \
  --output "$A/outputs/conditional_density_confirm_aggregate_${SLURM_JOB_ID}"
