#!/usr/bin/env bash
#SBATCH --job-name=llmtd-condembed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
EXTRA=()
if [[ "${COND_EMBED_FACTORIZED:-0}" == "1" ]]; then
  EXTRA+=(--factorized)
fi
cd "$REPO"
"$PY" "$ANALYSIS/scripts/embed_contract_conditions.py" \
  --env-file "$ROOT/EIPC/.env" \
  --model "${COND_EMBED_MODEL:-text-embedding-3-large}" \
  --dimensions "${COND_EMBED_DIM:-512}" \
  --output "${COND_EMBED_OUTPUT:-$ANALYSIS/outputs/contract_condition_embeddings_20260925}" \
  "${EXTRA[@]}"
