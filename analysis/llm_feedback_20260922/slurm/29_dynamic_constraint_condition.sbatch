#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dyncon
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${DYN_CONDITION:?missing condition}"
: "${DYN_TAG:?missing output tag}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/dynamic_constraint_sweep.py" \
  --cases "${DYN_CASES:-$RUN_ROOT/dynamic_constraint_bank_20260924/dynamic_cases.jsonl}" \
  --env-file "$ROOT/EIPC/.env" \
  --output "$RUN_ROOT/dynamic_constraint_${DYN_TAG}_${SLURM_JOB_ID}" \
  --condition "$DYN_CONDITION" \
  --model "${DYN_MODEL:-gpt-4.1-mini}" \
  --batch-size "${DYN_BATCH_SIZE:-10}"
