#!/usr/bin/env bash
#SBATCH --job-name=llmtd-feedback-topology
#SBATCH --partition=prioritized
#SBATCH --array=0-1
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
DATASETS=(shoppers default)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
seed=20261107

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

for topology in dynamic frozen frozen_state; do
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
    --graph "$graph" --evaluation-graph "$evaluation" --schedule "$schedule" \
    --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
    --samples 8192 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --graph-weight 1 \
    --max-actions-per-row-edge 4 --cooldown-steps 1 --schedule-mode common --common-cutoff .75 \
    --gate-mode current --topology-mode "$topology" --state-gate "$state_gate" \
    --active-min-t .15 --active-max-t .60 --numeric-guidance-mode feedback_adaptive \
    --numeric-guidance-scale 4 --feedback-repeat-growth 2 --feedback-late-multiplier 4 \
    --max-guidance-scale 24 --max-guidance-delta .75 --online-modes x0_reward_calibrated \
    --output "$A/outputs/feedback_topology_${dataset}_${topology}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done

hash_count=$(sha256sum "$A"/outputs/feedback_topology_${dataset}_*_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples/unconditional.csv | cut -d' ' -f1 | sort -u | wc -l)
test "$hash_count" -eq 1
