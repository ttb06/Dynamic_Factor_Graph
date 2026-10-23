#!/usr/bin/env bash
#SBATCH --job-name=llmtd-impladapt
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${IMPL_MODE:?missing filter, repair_preserve, or repair_gated}"
: "${IMPL_SEED:?missing seed}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/implication_condition_adapter.py" \
  --repo "$REPO" \
  --config "$RUN_ROOT/full_train_1028011/config.pkl" \
  --checkpoint "$RUN_ROOT/full_train_1028011/ema_model_epoch_8000.pt" \
  --output "$RUN_ROOT/implication_adapter_${IMPL_MODE}_${IMPL_SEED}_${SLURM_JOB_ID}" \
  --training-mode "$IMPL_MODE" \
  --epochs "${IMPL_EPOCHS:-100}" \
  --pairs-per-contract "${IMPL_PAIRS:-2048}" \
  --batch-size "${IMPL_BATCH:-2048}" \
  --eval-samples "${IMPL_EVAL:-512}" \
  --guidance-scales "${IMPL_GUIDANCE:-1,2}" \
  --sample-steps "${IMPL_STEPS:-30}" \
  --seed "$IMPL_SEED"
