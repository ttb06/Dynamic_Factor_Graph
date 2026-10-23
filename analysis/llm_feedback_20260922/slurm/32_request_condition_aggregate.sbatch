#!/usr/bin/env bash
#SBATCH --job-name=llmtd-reqagg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${REQ_JOB_IDS:?missing comma-separated request-condition job IDs}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/request_condition_aggregate.py" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/request_condition_aggregate_${SLURM_JOB_ID}" \
  --job-ids "$REQ_JOB_IDS" \
  --local-summary "${REQ_LOCAL_SUMMARY:-$RUN_ROOT/request_condition_local_20260924/summary.json}"
