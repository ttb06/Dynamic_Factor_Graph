#!/usr/bin/env bash
#SBATCH --job-name=llmtd-compaudit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${GROUND_JOB_41:?missing grounded GPT-4.1-mini job}"
: "${GROUND_JOB_54:?missing grounded GPT-5.4-mini job}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/compiler_counterexample_audit.py" \
  --cases "$RUN_ROOT/request_condition_shift_bank_20260925/cases.jsonl" \
  --requests "$RUN_ROOT/request_condition_shift_bank_20260925/requests.jsonl" \
  --vocabulary "$RUN_ROOT/adult_schema_vocabulary_20260925.json" \
  --compiled "$RUN_ROOT/request_condition_api_shift_41m_compile_1033605/compiled_requests.csv" \
  --compiled "$RUN_ROOT/request_condition_api_shift_54m_compile_1033607/compiled_requests.csv" \
  --compiled "$RUN_ROOT/request_condition_api_shift_ground_41m_compile_${GROUND_JOB_41}/compiled_requests.csv" \
  --compiled "$RUN_ROOT/request_condition_api_shift_ground_54m_compile_${GROUND_JOB_54}/compiled_requests.csv" \
  --output "$RUN_ROOT/compiler_counterexample_audit_${SLURM_JOB_ID}"
