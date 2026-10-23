#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xconstraint
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/tabdiff/bin/python"
TAG="${AUDIT_TAG:-cross_dataset_constraint_${SLURM_JOB_ID}}"
OUTPUT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}/$TAG"

mkdir -p "$OUTPUT"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/cross_dataset_constraint_audit.py" \
  --repo "$REPO" \
  --output "$OUTPUT" \
  --datasets adult default shoppers \
  --min-support "${MIN_SUPPORT:-50}" \
  --min-confidence "${MIN_CONFIDENCE:-0.80}" \
  --max-per-antecedent "${MAX_PER_ANTECEDENT:-8}"
