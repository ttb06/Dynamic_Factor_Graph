#!/usr/bin/env bash
#SBATCH --job-name=llmtd-prog-graph
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(default shoppers)
source_runs=(full_train_default_1033760 full_train_shoppers_1033761)
graph_indices=(0 1)
eval_indices=(1 0)
i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$i]}
source_run=${source_runs[$i]}
graph_index=${graph_indices[$i]}
eval_index=${eval_indices[$i]}
probe="$A/outputs/bscg_dynamic_trajectory_${dataset}_1080086_${i}"
graph="$A/outputs/bootstrap_conditional_graph_${dataset}_1070443_${graph_index}/graph.json"

cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" \
  --repo "$REPO" --dataset "$dataset" --graph "$graph" --final-graph "$graph" \
  --evaluation-graph "$A/outputs/actionable_union_eval_${dataset}_1060955_${eval_index}/evaluation_graph.json" \
  --schedule "$probe/typed_readability_schedule.csv" --trajectory-profile "$probe/edge_state_trajectory.csv" \
  --streak-profile "$probe/violation_streak_calibration.csv" \
  --config "$A/outputs/$source_run/config.pkl" --checkpoint "$A/outputs/$source_run/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261447 --max-actuation-calibration-rate .05 \
  --authority-mode graph_precursor --authority-min-surplus 0 \
  --online-modes x0_reward_calibrated --guided-final-repair \
  --schedule-mode edge --topology-mode row_ema_hysteresis --volatility-threshold .15 --volatility-ema .8 \
  --state-gate annealed8 --gate-mode persistent \
  --numeric-guidance-mode hardness_adaptive --max-actions-per-row-edge 2 \
  --cooldown-steps 2 --action-hold-steps 2 --max-guidance-delta 3.0 \
  --density-weight 1.0 --edit-weight .5 --gate-ranking trajectory_posterior \
  --active-min-t .00 --active-max-t .65 \
  --output "$A/outputs/progressive_graph_controller_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
