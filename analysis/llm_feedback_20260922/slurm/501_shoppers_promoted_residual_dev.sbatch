#!/usr/bin/env bash
#SBATCH --job-name=llmtd-shop-promote-resid
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json" \
  --evaluation-graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --schedule "$A/outputs/actionable_trajectory_v2_shoppers_1060954_0/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261382 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 1 --max-actions-per-row-edge 4 --cooldown-steps 1 \
  --schedule-mode common --common-cutoff .75 --gate-mode current --topology-mode dynamic \
  --state-gate current --active-min-t .15 --active-max-t .60 \
  --numeric-guidance-mode feedback_adaptive --numeric-guidance-scale 4 \
  --feedback-repeat-growth 2 --feedback-late-multiplier 4 --max-guidance-scale 24 \
  --max-guidance-delta 3 --promote-score-only --promoted-min-streak 3 \
  --promoted-max-calibration-rate .05 --online-modes x0_reward_calibrated \
  --action-planner edge_local --calibration-policy point --gate-ranking severity --guided-final-repair \
  --output "$A/outputs/shoppers_promoted_residual_dev_${SLURM_JOB_ID}"
