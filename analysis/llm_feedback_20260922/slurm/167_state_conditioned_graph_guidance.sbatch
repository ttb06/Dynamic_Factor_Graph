#!/usr/bin/env bash
#SBATCH --job-name=llmtd-state-guide
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(adult default shoppers)
CONFIGS=(
  "$A/outputs/full_train_1028011/config.pkl"
  "$A/outputs/full_train_default_1033760/config.pkl"
  "$A/outputs/full_train_shoppers_1033761/config.pkl"
)
CHECKPOINTS=(
  "$A/outputs/full_train_1028011/ema_model_epoch_8000.pt"
  "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"
  "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt"
)

dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
output="$A/outputs/state_graph_guidance_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/state_conditioned_graph_guidance.py" \
  --repo "$REPO" --dataset "$dataset" \
  --catalog "$A/outputs/cross_dataset_constraint_20260924/all_constraint_candidates.csv" \
  --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}" \
  --checkpoint "${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}" \
  --output "$output" --samples 2048 --steps 50 --max-edges 12 \
  --strength 1 --stable-fraction 0.70 --drift 0.25 --seed 20261004
