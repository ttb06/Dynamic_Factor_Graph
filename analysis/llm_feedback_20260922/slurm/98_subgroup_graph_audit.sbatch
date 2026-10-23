#!/usr/bin/env bash
#SBATCH --job-name=llmtd-subgraph
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/tabdiff/bin/python"
OUT="${ANALYSIS_RUN_ROOT:-$A/outputs}/subgroup_graph_audit_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/subgroup_graph_audit.py" \
  --repo "$REPO" \
  --candidate-file "$A/outputs/extended_constraint_audit_1048593/all_constraint_candidates.csv" \
  --output "$OUT" \
  --datasets adult default shoppers diabetes beijing news magic
