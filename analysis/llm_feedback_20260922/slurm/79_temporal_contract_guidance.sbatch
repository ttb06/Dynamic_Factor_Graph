#!/usr/bin/env bash
#SBATCH --job-name=llmtd-temporal
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${TEMP_DATASET:?}"; : "${TEMP_CATALOG:?}"; : "${TEMP_SEED:?}"; : "${TEMP_ADAPTER:?}"
: "${TEMP_PROFILE:?}"; : "${TEMP_PEAK:?}"; : "${TEMP_FLOOR:?}"; : "${TEMP_CUTOFF:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$TEMP_DATASET" in
  adult) D="$R/full_train_1028011" ;;
  default) D="$R/full_train_default_1033760" ;;
  shoppers) D="$R/full_train_shoppers_1033761" ;;
  *) exit 2 ;;
esac
TAG="${TEMP_PROFILE}_p${TEMP_PEAK}_f${TEMP_FLOOR}_c${TEMP_CUTOFF}"
TEMP_SCOPE="${TEMP_SCOPE:-all}"
TEMP_SCOPE_CUTOFF="${TEMP_SCOPE_CUTOFF:-0.25}"
TEMP_NEIGHBOR_K="${TEMP_NEIGHBOR_K:-2}"
TEMP_GRAPH_CANDIDATE_K="${TEMP_GRAPH_CANDIDATE_K:-8}"
OUT="$R/temporal_contract_guidance_${TEMP_DATASET}_${TEMP_CATALOG}_${TAG}_scope${TEMP_SCOPE}_sc${TEMP_SCOPE_CUTOFF}_k${TEMP_NEIGHBOR_K}_g${TEMP_GRAPH_CANDIDATE_K}_seed${TEMP_SEED}_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/temporal_contract_guidance.py" --repo "$REPO" --dataset "$TEMP_DATASET" \
  --catalog "$TEMP_CATALOG" --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" \
  --adapter "$TEMP_ADAPTER" --output "$OUT" --profile "$TEMP_PROFILE" --peak "$TEMP_PEAK" \
  --floor "$TEMP_FLOOR" --cutoff "$TEMP_CUTOFF" --scope "$TEMP_SCOPE" \
  --scope-cutoff "$TEMP_SCOPE_CUTOFF" \
  --neighbor-k "$TEMP_NEIGHBOR_K" \
  --graph-candidate-k "$TEMP_GRAPH_CANDIDATE_K" \
  --samples 2048 --steps 30 --seed "$TEMP_SEED"
