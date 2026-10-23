#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dyngraph
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-215
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"

DATASETS=(default shoppers)
CATALOGS=(support_numeric support_mixed)
PROFILES=(constant step_release midpulse)
PEAKS=(4 4 4)
FLOORS=(4 0 0)
ACTIVE_K=(1 2 4)
CANDIDATE_K=(4 8)
SEEDS=(20260925 20260926 20260927)

index=$SLURM_ARRAY_TASK_ID
seed_i=$((index % 3)); index=$((index / 3))
pool_i=$((index % 2)); index=$((index / 2))
active_i=$((index % 3)); index=$((index / 3))
profile_i=$((index % 3)); index=$((index / 3))
catalog_i=$((index % 2)); index=$((index / 2))
dataset_i=$((index % 2))

dataset=${DATASETS[$dataset_i]}
catalog=${CATALOGS[$catalog_i]}
profile=${PROFILES[$profile_i]}
peak=${PEAKS[$profile_i]}
floor=${FLOORS[$profile_i]}
active_k=${ACTIVE_K[$active_i]}
candidate_k=${CANDIDATE_K[$pool_i]}
seed=${SEEDS[$seed_i]}

case "$dataset" in
  default)
    D="$R/full_train_default_1033760"
    if [[ "$catalog" == support_numeric ]]; then
      case "$seed" in
        20260925) ADAPTER="$R/cross_condition_default_structured_seed20260925_1044270/adapter.pt" ;;
        20260926) ADAPTER="$R/cross_condition_default_structured_seed20260926_1044271/adapter.pt" ;;
        20260927) ADAPTER="$R/cross_condition_default_structured_seed20260927_1044272/adapter.pt" ;;
      esac
    else
      case "$seed" in
        20260925) ADAPTER="$R/method_adapter_dense_full_default_seed20260925_1045844/adapter.pt" ;;
        20260926) ADAPTER="$R/method_adapter_dense_full_default_seed20260926_1045847/adapter.pt" ;;
        20260927) ADAPTER="$R/method_adapter_dense_full_default_seed20260927_1045850/adapter.pt" ;;
      esac
    fi
    ;;
  shoppers)
    D="$R/full_train_shoppers_1033761"
    if [[ "$catalog" == support_numeric ]]; then
      case "$seed" in
        20260925) ADAPTER="$R/cross_condition_shoppers_structured_seed20260925_1044276/adapter.pt" ;;
        20260926) ADAPTER="$R/cross_condition_shoppers_structured_seed20260926_1044277/adapter.pt" ;;
        20260927) ADAPTER="$R/cross_condition_shoppers_structured_seed20260927_1044278/adapter.pt" ;;
      esac
    else
      case "$seed" in
        20260925) ADAPTER="$R/method_adapter_dense_full_shoppers_seed20260925_1045898/adapter.pt" ;;
        20260926) ADAPTER="$R/method_adapter_dense_full_shoppers_seed20260926_1045901/adapter.pt" ;;
        20260927) ADAPTER="$R/method_adapter_dense_full_shoppers_seed20260927_1045904/adapter.pt" ;;
      esac
    fi
    ;;
esac

tag="${profile}_p${peak}_f${floor}_c0.5"
OUT="$R/temporal_contract_guidance_${dataset}_${catalog}_${tag}_scopedynamic_graph_sc0.25_k${active_k}_g${candidate_k}_seed${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/temporal_contract_guidance.py" --repo "$REPO" --dataset "$dataset" \
  --catalog "$catalog" --config "$D/config.pkl" --checkpoint "$D/ema_model_epoch_8000.pt" \
  --adapter "$ADAPTER" --output "$OUT" --profile "$profile" --peak "$peak" \
  --floor "$floor" --cutoff 0.5 --scope dynamic_graph --scope-cutoff 0.25 \
  --neighbor-k "$active_k" --graph-candidate-k "$candidate_k" \
  --samples 2048 --steps 30 --seed "$seed"
