#!/usr/bin/env bash
#SBATCH --job-name=llmtd-semllm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${SEM_BANK_JOB_ID:?missing bank job id}"
: "${SEM_CONDITION:?missing condition}"
: "${SEM_TAG:?missing output tag}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/semantic_llm_sweep.py" \
  --cases "$RUN_ROOT/semantic_bank_${SEM_BANK_JOB_ID}/semantic_cases.jsonl" \
  --env-file "$ROOT/EIPC/.env" \
  --output "$RUN_ROOT/semantic_llm_${SEM_TAG}_${SLURM_JOB_ID}" \
  --condition "$SEM_CONDITION" \
  --model "${SEM_MODEL:-gpt-4.1-mini}" \
  --batch-size "${SEM_BATCH_SIZE:-10}" \
  --repeats "${SEM_REPEATS:-1}" \
  --inter-call-delay "${SEM_DELAY:-3}"
