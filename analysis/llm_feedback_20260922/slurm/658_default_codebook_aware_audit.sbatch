#!/usr/bin/env bash
#SBATCH --job-name=llmtd-codebook-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${GRAPH_JOB_ID:?GRAPH_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/audit_constraint_meaningfulness.py" \
  --run "$A/outputs/default_codebook_aware_${GRAPH_JOB_ID}" \
  --repo "$REPO" --dataset default \
  --output "$A/outputs/default_codebook_aware_audit_${SLURM_JOB_ID}"
