#!/usr/bin/env bash
#SBATCH --job-name=llmtd-adaptvar
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${VAR_DATASET:?}"; : "${VAR_SEED:?}"; : "${VAR_LABEL:?}"; : "${VAR_CORE_LR:?}"
: "${VAR_CURRICULUM:?}"; : "${VAR_DROPOUT:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
if [[ -n "${VAR_BACKBONE_DIR:-}" ]]; then
  D="$VAR_BACKBONE_DIR"
else
  case "$VAR_DATASET" in
    adult) D="$R/full_train_1028011" ;;
    default) D="$R/full_train_default_1033760" ;;
    shoppers) D="$R/full_train_shoppers_1033761" ;;
    *) exit 2 ;;
  esac
fi
VAR_REPRESENTATION="${VAR_REPRESENTATION:-structured}"
OUT="$R/method_adapter_${VAR_LABEL}_${VAR_REPRESENTATION}_${VAR_DATASET}_seed${VAR_SEED}_${SLURM_JOB_ID}"
EMBED_ARGS=()
if [[ -n "${VAR_EMBEDDINGS:-}" ]]; then EMBED_ARGS+=(--embeddings "$VAR_EMBEDDINGS"); fi
cd "$REPO"
"$PY" "$A/scripts/mixed_type_condition_adapter.py" --repo "$REPO" --dataset "$VAR_DATASET" \
  --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" --output "$OUT" \
  --representation "$VAR_REPRESENTATION" "${EMBED_ARGS[@]}" \
  --epochs 100 --pairs-per-contract 1536 --eval-samples 512 \
  --guidance-scales 1:2 --adapter-lr 5e-4 --core-lr "$VAR_CORE_LR" \
  --condition-dropout "$VAR_DROPOUT" --threshold-curriculum "$VAR_CURRICULUM" \
  --run-label "$VAR_LABEL" --seed "$VAR_SEED"
