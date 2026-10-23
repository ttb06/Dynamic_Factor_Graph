#!/usr/bin/env bash
#SBATCH --job-name=llmtd-diab-prog
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${PROBE_JOB_ID:?PROBE_JOB_ID required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
probe="$A/outputs/bscg_dynamic_trajectory_diabetes_${PROBE_JOB_ID}"
graph="$A/outputs/bootstrap_conditional_graph_diabetes_1070443_2/graph.json"
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" \
  --repo "$REPO" --dataset diabetes --graph "$graph" --final-graph "$graph" \
  --evaluation-graph "$A/outputs/actionable_union_eval_diabetes_1060955_2/evaluation_graph.json" \
  --schedule "$probe/typed_readability_schedule.csv" --trajectory-profile "$probe/edge_state_trajectory.csv" \
  --streak-profile "$probe/violation_streak_calibration.csv" \
  --config "$A/outputs/full_train_diabetes_1048479/config.pkl" \
  --checkpoint "$A/outputs/full_train_diabetes_1048479/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261447 --max-actuation-calibration-rate .05 \
  --authority-mode graph_precursor --authority-min-surplus 0 \
  --online-modes x0_reward_calibrated --guided-final-repair \
  --schedule-mode edge --topology-mode row_ema_hysteresis --volatility-threshold .15 --volatility-ema .8 \
  --state-gate annealed8 --gate-mode persistent \
  --numeric-guidance-mode hardness_adaptive --max-actions-per-row-edge 2 \
  --cooldown-steps 2 --action-hold-steps 2 --max-guidance-delta 3.0 \
  --density-weight 1.0 --edit-weight .5 --gate-ranking trajectory_posterior \
  --active-min-t .00 --active-max-t .65 \
  --output "$A/outputs/progressive_graph_controller_diabetes_${SLURM_JOB_ID}"
