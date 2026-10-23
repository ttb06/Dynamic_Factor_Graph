#!/usr/bin/env bash
#SBATCH --job-name=llmtd-condadapt
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${COND_REP:?missing condition representation}"
: "${COND_TAG:?missing output tag}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
EXTRA=()
if [[ "$COND_REP" == "text" || "$COND_REP" == "hybrid" ]]; then
  EXTRA+=(--embeddings "${COND_EMBEDDINGS:-$RUN_ROOT/contract_condition_embeddings_20260925/embeddings.json}")
fi
cd "$REPO"
"$PY" "$ANALYSIS/scripts/conditioned_tabdiff_adapter.py" \
  --repo "$REPO" \
  --config "$RUN_ROOT/full_train_1028011/config.pkl" \
  --checkpoint "$RUN_ROOT/full_train_1028011/ema_model_epoch_8000.pt" \
  --output "$RUN_ROOT/condition_adapter_${COND_TAG}_${SLURM_JOB_ID}" \
  --representation "$COND_REP" \
  --epochs "${COND_EPOCHS:-100}" \
  --pairs-per-contract "${COND_PAIRS:-2048}" \
  --batch-size "${COND_BATCH:-2048}" \
  --eval-samples "${COND_EVAL:-256}" \
  --guidance-scales "${COND_GUIDANCE:-1,2}" \
  --sample-steps "${COND_STEPS:-30}" \
  --text-branch-dropout "${COND_TEXT_DROPOUT:-0}" \
  --monotonic-weight "${COND_MONO_WEIGHT:-0}" \
  --monotonic-margin "${COND_MONO_MARGIN:-0.05}" \
  --monotonic-batch "${COND_MONO_BATCH:-256}" \
  --seed "${COND_SEED:-20260925}" \
  "${EXTRA[@]}"
