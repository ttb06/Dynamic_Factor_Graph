#!/usr/bin/env bash
#SBATCH --job-name=llmtd-sparse-scope
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="$A/outputs"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
DATASETS=(default default default shoppers shoppers shoppers)
SEEDS=(20260925 20260926 20260927 20260925 20260926 20260927)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
if [[ "$DATASET" == default ]]; then
  CONFIG="$R/full_train_default_1033760/config.pkl"
  CHECKPOINT="$R/full_train_default_1033760/ema_model_epoch_8000.pt"
  ADAPTER="$R/subgroup_amount_adapter_1050333/adapter.pt"
else
  CONFIG="$R/full_train_shoppers_1033761/config.pkl"
  CHECKPOINT="$R/full_train_shoppers_1033761/ema_model_epoch_8000.pt"
  ADAPTER="$R/method_adapter_dense_full_shoppers_seed20260925_1045898/adapter.pt"
fi
OUT="$R/sparse_scope_${DATASET}_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/subgroup_conditioned_sampling.py" --repo "$REPO" --dataset "$DATASET" \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --adapter "$ADAPTER" --output "$OUT" \
  --samples 2048 --steps 30 --seed "$SEED" --guidances 2 --focused
