#!/usr/bin/env bash
#SBATCH --job-name=llmtd-rowedge-dual
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(default default default shoppers shoppers shoppers)
variants=(static_window dynamic_hysteresis primal_dual static_window dynamic_hysteresis primal_dual)
source_runs=(full_train_default_1033760 full_train_default_1033760 full_train_default_1033760 full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_shoppers_1033761)
graph_indices=(0 0 0 1 1 1); probe_indices=(0 0 0 1 1 1); eval_indices=(1 1 1 0 0 0)
i=$SLURM_ARRAY_TASK_ID; dataset=${datasets[$i]}; variant=${variants[$i]}; source_run=${source_runs[$i]}
graph_index=${graph_indices[$i]}; probe_index=${probe_indices[$i]}; eval_index=${eval_indices[$i]}
probe="$A/outputs/bscg_dynamic_trajectory_${dataset}_${TRAJECTORY_JOB_ID}_${probe_index}"
common=(
  --repo "$REPO" --dataset "$dataset"
  --graph "$A/outputs/bootstrap_conditional_graph_${dataset}_1070443_${graph_index}/graph.json"
  --evaluation-graph "$A/outputs/actionable_union_eval_${dataset}_1060955_${eval_index}/evaluation_graph.json"
  --schedule "$probe/typed_readability_schedule.csv"
  --trajectory-profile "$probe/edge_state_trajectory.csv"
  --streak-profile "$probe/violation_streak_calibration.csv"
  --config "$A/outputs/$source_run/config.pkl" --checkpoint "$A/outputs/$source_run/ema_model_epoch_8000.pt"
  --samples 4096 --steps 50 --seed 20261447 --max-actuation-calibration-rate .05
  --online-modes x0_reward_calibrated --guided-final-repair
  --density-weight 1.0 --edit-weight .5 --gate-ranking trajectory_posterior
  --active-min-t .00 --active-max-t .65 --max-guidance-delta 3.0
)
case "$variant" in
  static_window)
    policy=(--schedule-mode common --common-cutoff .40 --topology-mode frozen_state \
      --state-gate stable --numeric-guidance-mode hardness_adaptive \
      --max-actions-per-row-edge 1 --cooldown-steps 2)
    ;;
  dynamic_hysteresis)
    policy=(--schedule-mode edge --topology-mode row_ema_hysteresis --volatility-threshold .15 \
      --volatility-ema .8 --state-gate stable --numeric-guidance-mode hardness_adaptive \
      --max-actions-per-row-edge 2 --cooldown-steps 2 --action-hold-steps 2)
    ;;
  primal_dual)
    policy=(--schedule-mode edge --topology-mode row_ema_hysteresis --volatility-threshold .15 \
      --volatility-ema .8 --state-gate dual --numeric-guidance-mode primal_dual \
      --numeric-guidance-scale 1.0 --max-guidance-scale 6.0 --dual-lr .60 --dual-decay .85 \
      --dual-threshold 1.0 --dual-max 6.0 --max-actions-per-row-edge 3 --cooldown-steps 1 \
      --action-hold-steps 2)
    ;;
esac
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" "${common[@]}" "${policy[@]}" \
  --output "$A/outputs/rowedge_dynamic_${dataset}_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
