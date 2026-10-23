#!/usr/bin/env bash
#SBATCH --job-name=llmtd-action-hold
#SBATCH --partition=prioritized
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers default diabetes); HOLDS=(0 2 4 8)
dataset=${DATASETS[$((SLURM_ARRAY_TASK_ID / 4))]}; hold=${HOLDS[$((SLURM_ARRAY_TASK_ID % 4))]}
if [[ "$dataset" == shoppers ]]; then
  run=full_train_shoppers_1033761; catalog=expanded_constraint_catalog_1057680
  trajectory=expanded_catalog_trajectory_1057681; samples=8192; state_gate=current
elif [[ "$dataset" == default ]]; then
  run=full_train_default_1033760; catalog=cross_dataset_catalog_default_1058707_1
  trajectory=cross_dataset_trajectory_default_1058708_1; samples=8192; state_gate=stable5
else
  run=full_train_diabetes_1048479; catalog=diabetes_catalog_1060368
  trajectory=diabetes_trajectory_1060369; samples=4096; state_gate=current
fi
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/$catalog/constraint_catalog.json" \
  --schedule "$A/outputs/$trajectory/typed_readability_schedule.csv" \
  --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
  --samples "$samples" --steps 50 --seed 20261087 --min-support 24 --drift .10 \
  --density-weight 1.0 --edit-weight .5 --graph-weight 0 \
  --max-actions-per-row-edge 1 --cooldown-steps 3 --action-hold-steps "$hold" \
  --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode dynamic \
  --state-gate "$state_gate" --active-min-t .25 --active-max-t .50 \
  --numeric-guidance-mode constant --numeric-guidance-scale 1 \
  --online-modes x0_reward_calibrated \
  --output "$A/outputs/action_hold_${dataset}_h${hold}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
