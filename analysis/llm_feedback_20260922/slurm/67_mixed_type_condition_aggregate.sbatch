#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixagg
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"; R="${ANALYSIS_RUN_ROOT:-$A/outputs}"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"; "$PY" "$A/scripts/mixed_type_condition_aggregate.py" --input-root "$R" --glob "${MIXED_GLOB:-mixed_condition_*_seed*/contract_metrics.csv}" --output "$R/${MIXED_AGG_TAG:-mixed_condition_aggregate}_${SLURM_JOB_ID}"
