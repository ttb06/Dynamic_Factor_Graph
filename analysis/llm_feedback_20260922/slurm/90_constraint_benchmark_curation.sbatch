#!/usr/bin/env bash
#SBATCH --job-name=llmtd-curate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
CURATION_CANDIDATES="${CURATION_CANDIDATES:-$R/extended_constraint_audit_1048475/all_constraint_candidates.csv}"
"$PY" "$A/scripts/constraint_benchmark_curation.py" \
  --candidates "$CURATION_CANDIDATES" \
  --output "$R/constraint_benchmark_curation_${SLURM_JOB_ID}"
