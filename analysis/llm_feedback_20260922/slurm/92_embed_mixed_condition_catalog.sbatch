#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixembed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${EMBED_DATASET:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
EXTRA=(); if [[ "${EMBED_FACTORIZED:-0}" == 1 ]]; then EXTRA+=(--factorized); fi
OUT="$R/mixed_condition_embeddings_${EMBED_DATASET}_${EMBED_TAG:-raw}_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/embed_mixed_condition_catalog.py" --dataset "$EMBED_DATASET" \
  --env-file "$ROOT/EIPC/.env" --output "$OUT" --model "${EMBED_MODEL:-text-embedding-3-large}" \
  --dimensions "${EMBED_DIMENSIONS:-256}" "${EXTRA[@]}"
