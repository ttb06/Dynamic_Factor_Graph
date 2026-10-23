#!/usr/bin/env bash
#SBATCH --job-name=llmtd-typeaudit
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"; R="${ANALYSIS_RUN_ROOT:-$A/outputs}"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"; "$PY" "$A/scripts/compiler_typecheck_property_audit.py" --runs "$R/request_condition_aggregate_1044443/runs.csv" --bank-root "$R" --output "$R/compiler_typecheck_property_${SLURM_JOB_ID}"
