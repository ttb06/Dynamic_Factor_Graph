#!/usr/bin/env bash
#SBATCH --job-name=llmtd-reqcond
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${REQ_MODE:?missing direct or compile mode}"
: "${REQ_TAG:?missing output tag}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
EXTRA=()
if [[ -n "${REQ_VOCAB:-}" ]]; then
  EXTRA+=(--categorical-vocabulary "$REQ_VOCAB")
fi
if [[ -n "${REQ_SCHEMA:-}" ]]; then
  EXTRA+=(--schema-file "$REQ_SCHEMA")
fi
cd "$REPO"
"$PY" "$ANALYSIS/scripts/request_condition_probe.py" \
  --cases "${REQ_CASES:-$RUN_ROOT/request_condition_bank_20260924/cases.jsonl}" \
  --requests "${REQ_REQUESTS:-$RUN_ROOT/request_condition_bank_20260924/requests.jsonl}" \
  --env-file "$ROOT/EIPC/.env" \
  --output "$RUN_ROOT/request_condition_api_${REQ_TAG}_${SLURM_JOB_ID}" \
  --mode "$REQ_MODE" \
  --model "${REQ_MODEL:-gpt-4.1-mini}" \
  --batch-size "${REQ_BATCH_SIZE:-10}" \
  --compile-strategy "${REQ_COMPILE_STRATEGY:-basic}" \
  --seed "${REQ_SEED:-20260924}" \
  "${EXTRA[@]}"
