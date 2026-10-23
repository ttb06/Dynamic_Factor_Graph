#!/usr/bin/env bash
#SBATCH --job-name=llmtd-strong-topology
#SBATCH --partition=prioritized
#SBATCH --array=0-3
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
DATASETS=(shoppers shoppers default default); SEEDS=(20261090 20261091 20261090 20261091)
i=$SLURM_ARRAY_TASK_ID; dataset=${DATASETS[$i]}; seed=${SEEDS[$i]}
if [[ "$dataset" == shoppers ]]; then
  graph="$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"
  schedule="$A/outputs/actionable_trajectory_v2_shoppers_1060954_0/typed_readability_schedule.csv"
  run=full_train_shoppers_1033761; state_gate=current
else
  graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  schedule="$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv"
  run=full_train_default_1033760; state_gate=stable5
fi
for topology in dynamic frozen_state; do
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
    --graph "$graph" --evaluation-graph "$evaluation" --schedule "$schedule" \
    --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
    --samples 8192 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --graph-weight 1 \
    --max-actions-per-row-edge 1 --cooldown-steps 3 --schedule-mode common --common-cutoff .50 \
    --gate-mode current --topology-mode "$topology" --state-gate "$state_gate" \
    --active-min-t .25 --active-max-t .50 --numeric-guidance-mode constant --numeric-guidance-scale 4 \
    --online-modes x0_reward_calibrated \
    --output "$A/outputs/strong_topology_${dataset}_${topology}_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
done
first="$A/outputs/strong_topology_${dataset}_dynamic_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}/samples/unconditional.csv"
second="$A/outputs/strong_topology_${dataset}_frozen_state_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}/samples/unconditional.csv"
test "$(sha256sum "$first" | cut -d' ' -f1)" = "$(sha256sum "$second" | cut -d' ' -f1)"
