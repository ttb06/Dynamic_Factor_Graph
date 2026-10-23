#!/usr/bin/env bash
#SBATCH --job-name=llmtd-shop-dyn-confirm
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
seed_index=$((i / 2))
topology_index=$((i % 2))
seeds=(20261380 20261381)
topologies=(dynamic frozen_state)
seed=${seeds[$seed_index]}
topology=${topologies[$topology_index]}

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/trajectory_portfolio_v2_shoppers_1060979_0/constraint_catalog.json" \
  --final-graph "$A/outputs/trajectory_portfolio_v2_shoppers_1060979_0/constraint_catalog.json" \
  --evaluation-graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --schedule "$A/outputs/e2e_graph_trajectory_pretrained_1062231_0/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 0 \
  --max-actions-per-row-edge 4 --cooldown-steps 1 --schedule-mode edge --gate-mode current \
  --topology-mode "$topology" --state-gate current --active-min-t .15 --active-max-t .60 \
  --numeric-guidance-mode feedback_adaptive --numeric-guidance-scale 2 \
  --feedback-repeat-growth 1.5 --feedback-late-multiplier 2 --max-guidance-scale 8 \
  --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 4 --lock-applicability-columns \
  --calibration-policy split_median --calibration-z 0 --gate-ranking graph_rate --guided-final-repair \
  --output "$A/outputs/shoppers_dynamic_confirm_s${seed}_${topology}_${SLURM_ARRAY_JOB_ID}_${i}"
