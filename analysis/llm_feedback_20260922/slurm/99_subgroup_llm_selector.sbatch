#!/usr/bin/env bash
#SBATCH --job-name=llmtd-subgraph-llm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
MODELS=(gpt-4.1-mini gpt-5.4-mini)
TAGS=(gpt41mini gpt54mini)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
AUDIT_ID=${SUBGROUP_AUDIT_ID:-1050317}
OUT="${ANALYSIS_RUN_ROOT:-$A/outputs}/subgroup_llm_${TAG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/subgroup_llm_selector.py" \
  --audit "$A/outputs/subgroup_graph_audit_${AUDIT_ID}" \
  --env-file "$ROOT/EIPC/.env" \
  --output "$OUT" \
  --model "$MODEL"
