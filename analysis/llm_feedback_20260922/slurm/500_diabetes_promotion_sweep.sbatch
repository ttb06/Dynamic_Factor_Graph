#!/usr/bin/env bash
#SBATCH --job-name=llmtd-diab-promote-sweep
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-8
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
rates=(.01 .02 .05 .01 .02 .05 .01 .02 .05)
streaks=(2 2 2 3 3 3 5 5 5)
rate=${rates[$i]}; streak=${streaks[$i]}

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset diabetes \
  --graph "$A/outputs/trajectory_portfolio_v2_diabetes_1060979_2/constraint_catalog.json" \
  --evaluation-graph "$A/outputs/actionable_union_eval_diabetes_1060955_2/evaluation_graph.json" \
  --schedule "$A/outputs/actionable_trajectory_v2_diabetes_1060954_2/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_diabetes_1048479/config.pkl" \
  --checkpoint "$A/outputs/full_train_diabetes_1048479/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261382 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 1 \
  --max-actions-per-row-edge 4 --cooldown-steps 1 --schedule-mode common --common-cutoff .75 \
  --gate-mode current --topology-mode dynamic --state-gate current --active-min-t .05 --active-max-t .60 \
  --numeric-guidance-mode feedback_adaptive --numeric-guidance-scale 2 \
  --feedback-repeat-growth 2 --feedback-late-multiplier 4 --max-guidance-scale 16 \
  --promote-score-only --promoted-min-streak "$streak" --promoted-max-calibration-rate "$rate" \
  --online-modes x0_reward_calibrated --action-planner component_beam --component-beam-width 4 \
  --lock-applicability-columns --calibration-policy point --gate-ranking graph_rate \
  --output "$A/outputs/diabetes_promote_sweep_r${rate}_s${streak}_${SLURM_ARRAY_JOB_ID}_${i}"
