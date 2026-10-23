#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cal-adapt-confirm
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(default shoppers)
CONFIGS=("$A/outputs/full_train_default_1033760/config.pkl" "$A/outputs/full_train_shoppers_1033761/config.pkl")
CHECKPOINTS=("$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt")
TRAJECTORIES=("$A/outputs/general_state_graph_default_1054114_1/edge_state_trajectory.csv" "$A/outputs/general_state_graph_shoppers_1054114_2/edge_state_trajectory.csv")
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
graph=$(find "$A/outputs" -maxdepth 2 -path "*/unified_verified_graph_${dataset}_1054228_*/*unified_graph.json" | head -1)
test -n "$graph"
output="$A/outputs/calibrated_adaptive_confirm_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/state_conditioned_graph_guidance.py" --repo "$REPO" --dataset "$dataset" \
  --catalog "$A/outputs/cross_dataset_constraint_20260924/all_constraint_candidates.csv" --unified-graph "$graph" \
  --trajectory-calibration "${TRAJECTORIES[$SLURM_ARRAY_TASK_ID]}" \
  --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}" --checkpoint "${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}" \
  --output "$output" --samples 4096 --steps 50 --max-edges 24 --actuation-level hard_near \
  --strength 1 --stable-fraction 0.85 --drift 0.10 --readability-precision 0.90 \
  --readability-recall 0.80 --fallback-cutoff 0.25 --seed 20261005
