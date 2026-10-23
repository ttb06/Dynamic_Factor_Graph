#!/usr/bin/env bash
#SBATCH --job-name=llmtd-rarestat
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${RARE_AGG_ID:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/rare_support_statistical_audit.py" \
  --aggregate "$R/rejection_frontier_aggregate_${RARE_AGG_ID}" \
  --output "$R/rare_support_statistical_audit_${RARE_AGG_ID}_${SLURM_JOB_ID}" \
  --bootstrap-repeats 20000
