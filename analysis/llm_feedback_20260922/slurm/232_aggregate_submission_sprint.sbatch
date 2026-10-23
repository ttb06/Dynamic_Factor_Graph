#!/usr/bin/env bash
#SBATCH --job-name=llmtd-sprint-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${MAIN_JOB_ID:?}"; : "${WINDOW_JOB_ID:?}"; : "${QUALITY_JOB_ID:?}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/aggregate_submission_sprint.py" --analysis "$A" \
  --main-job-id "$MAIN_JOB_ID" --window-job-id "$WINDOW_JOB_ID" --quality-job-id "$QUALITY_JOB_ID" \
  --output "$A/outputs/submission_sprint_aggregate_${SLURM_JOB_ID}"
