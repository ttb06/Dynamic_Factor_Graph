#!/usr/bin/env bash
#SBATCH --job-name=llmtd-component-cross
#SBATCH --partition=prioritized
#SBATCH --array=0-3
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
datasets=(default default default diabetes)
scales=(4 4 2 1)
guards=(1 5 1 1)
profiles=(default_s4_g1 default_s4_g5 default_s2_g1 diabetes_s1_g1)
i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$i]}

if [[ "$dataset" == default ]]; then
  graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  schedule="$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv"
  run=full_train_default_1033760; state_gate=stable5; samples=8192; common_cutoff=.50
  active_min=.25; active_max=.50; max_actions=1; cooldown=3; seed=20261093
else
  graph="$A/outputs/trajectory_portfolio_v2_diabetes_1060979_2/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_diabetes_1060955_2/evaluation_graph.json"
  schedule="$A/outputs/actionable_trajectory_v2_diabetes_1060954_2/typed_readability_schedule.csv"
  run=full_train_diabetes_1048479; state_gate=current; samples=4096; common_cutoff=.50
  active_min=.25; active_max=.50; max_actions=1; cooldown=3; seed=20261093
fi

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$graph" --evaluation-graph "$evaluation" --schedule "$schedule" \
  --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
  --samples "$samples" --steps 50 --seed "$seed" --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight "${guards[$i]}" \
  --max-actions-per-row-edge "$max_actions" --cooldown-steps "$cooldown" \
  --schedule-mode common --common-cutoff "$common_cutoff" --gate-mode current \
  --topology-mode dynamic --state-gate "$state_gate" --active-min-t "$active_min" --active-max-t "$active_max" \
  --numeric-guidance-mode constant --numeric-guidance-scale "${scales[$i]}" \
  --max-guidance-scale 8 --online-modes x0_reward_calibrated \
  --action-planner component_beam --component-beam-width 4 --component-violation-weight 100 \
  --guided-final-repair \
  --output "$A/outputs/component_cross_${profiles[$i]}_${SLURM_ARRAY_JOB_ID}_${i}"
