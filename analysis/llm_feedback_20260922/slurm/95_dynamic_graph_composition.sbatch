#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dyncompose
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"

DATASETS=(default shoppers)
PROFILES=(constant midpulse)
SCOPES=(target_neighbors dynamic_graph)
index=$SLURM_ARRAY_TASK_ID
scope_i=$((index % 2)); index=$((index / 2))
profile_i=$((index % 2)); index=$((index / 2))
dataset_i=$((index % 2))
dataset=${DATASETS[$dataset_i]}
profile=${PROFILES[$profile_i]}
scope=${SCOPES[$scope_i]}
seed=20260925

case "$dataset" in
  default)
    D="$R/full_train_default_1033760"
    ADAPTER="$R/method_adapter_dense_full_default_seed20260925_1045844/adapter.pt"
    ;;
  shoppers)
    D="$R/full_train_shoppers_1033761"
    ADAPTER="$R/method_adapter_dense_full_shoppers_seed20260925_1045898/adapter.pt"
    ;;
esac
if [[ "$profile" == constant ]]; then floor=4; else floor=0; fi
OUT="$R/temporal_contract_guidance_${dataset}_mixed_${profile}_p4_f${floor}_c0.5_scope${scope}_sc0.25_k2_g4_seed${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/temporal_contract_guidance.py" --repo "$REPO" --dataset "$dataset" \
  --catalog mixed --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" \
  --adapter "$ADAPTER" --output "$OUT" --profile "$profile" --peak 4 --floor "$floor" \
  --cutoff 0.5 --scope "$scope" --scope-cutoff 0.25 --neighbor-k 2 \
  --graph-candidate-k 4 --samples 2048 --steps 30 --seed "$seed"
