#!/usr/bin/env bash
#SBATCH --job-name=llmtd-semagg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${SEM_LLM_JOB_IDS:?missing semantic LLM job IDs}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/semantic_sweep_aggregate.py" \
  --root "$RUN_ROOT" \
  --job-ids "$SEM_LLM_JOB_IDS" \
  --output "$RUN_ROOT/semantic_aggregate_${SLURM_JOB_ID}"
