#!/usr/bin/env bash
#SBATCH --job-name=llmtd-transition-policy
#SBATCH --partition=prioritized
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers shoppers shoppers default default default diabetes diabetes diabetes)
SEEDS=(20261071 20261072 20261073 20261071 20261072 20261073 20261071 20261072 20261073)
i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}
seed=${SEEDS[$i]}

if [[ "$dataset" == shoppers ]]; then
  run=full_train_shoppers_1033761
  catalog=expanded_constraint_catalog_1057680
  trajectory=expanded_catalog_trajectory_1057681
  samples=8192
elif [[ "$dataset" == default ]]; then
  run=full_train_default_1033760
  catalog=cross_dataset_catalog_default_1058707_1
  trajectory=cross_dataset_trajectory_default_1058708_1
  samples=8192
else
  run=full_train_diabetes_1048479
  catalog=diabetes_catalog_1060368
  trajectory=diabetes_trajectory_1060369
  samples=4096
fi

cd "$REPO"
for policy in immediate wait3 dynamic_wait5 topology_wait5 frozen_wait5; do
  topology=dynamic
  persistence=2
  state_gate=current
  if [[ "$policy" == wait3 ]]; then
    topology=hysteresis
    persistence=3
    state_gate=stable3
  elif [[ "$policy" == dynamic_wait5 ]]; then
    topology=dynamic
    state_gate=stable5
  elif [[ "$policy" == topology_wait5 ]]; then
    topology=hysteresis
    persistence=5
    state_gate=stable5
  elif [[ "$policy" == frozen_wait5 ]]; then
    topology=frozen_state
    state_gate=stable5
  fi
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
    --graph "$A/outputs/$catalog/constraint_catalog.json" \
    --schedule "$A/outputs/$trajectory/typed_readability_schedule.csv" \
    --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
    --samples "$samples" --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --cooldown-steps 3 \
    --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode "$topology" \
    --topology-persistence "$persistence" --state-gate "$state_gate" \
    --active-min-t .25 --active-max-t .50 --online-modes x0_reward_calibrated \
    --output "$A/outputs/transition_policy_${dataset}_${policy}_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done
