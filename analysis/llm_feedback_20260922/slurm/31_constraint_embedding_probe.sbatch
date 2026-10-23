#!/usr/bin/env bash
#SBATCH --job-name=llmtd-embprobe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${EMB_REPRESENTATION:?missing representation}"
: "${EMB_TAG:?missing tag}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/constraint_embedding_probe.py" \
  --cases "$RUN_ROOT/dynamic_constraint_bank_20260924/dynamic_cases.jsonl" \
  --env-file "$ROOT/EIPC/.env" \
  --output "$RUN_ROOT/constraint_embedding_${EMB_TAG}_${SLURM_JOB_ID}" \
  --model "${EMB_MODEL:-text-embedding-3-small}" \
  --representation "$EMB_REPRESENTATION" \
  --batch-size "${EMB_BATCH_SIZE:-100}"
