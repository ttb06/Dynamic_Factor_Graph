#!/usr/bin/env bash
#SBATCH --job-name=llmtd-ext-prog
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${DATASET:?}"; : "${GRAPH_JOB_ID:?}"; : "${TRAIN_JOB_ID:?}"; : "${PROBE_JOB_ID:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
graph="$A/outputs/extended_bscg_${DATASET}_${GRAPH_JOB_ID}/graph.json"
probe="$A/outputs/extended_dynamic_trajectory_${DATASET}_${PROBE_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" \
  --repo "$REPO" --dataset "$DATASET" --graph "$graph" --final-graph "$graph" --evaluation-graph "$graph" \
  --schedule "$probe/typed_readability_schedule.csv" --trajectory-profile "$probe/edge_state_trajectory.csv" \
  --streak-profile "$probe/violation_streak_calibration.csv" \
  --config "$A/outputs/extended_full_train_${DATASET}_${TRAIN_JOB_ID}/config.pkl" \
  --checkpoint "$A/outputs/extended_full_train_${DATASET}_${TRAIN_JOB_ID}/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261447 --max-actuation-calibration-rate .05 \
  --authority-mode graph_precursor --authority-min-surplus 0 \
  --online-modes x0_reward_calibrated --guided-final-repair \
  --schedule-mode edge --topology-mode row_ema_hysteresis --volatility-threshold .15 --volatility-ema .8 \
  --state-gate annealed8 --gate-mode persistent \
  --numeric-guidance-mode hardness_adaptive --max-actions-per-row-edge 2 \
  --cooldown-steps 2 --action-hold-steps 2 --max-guidance-delta 3.0 \
  --density-weight 1.0 --edit-weight .5 --gate-ranking trajectory_posterior \
  --active-min-t .00 --active-max-t .65 \
  --output "$A/outputs/extended_progressive_controller_${DATASET}_${SLURM_JOB_ID}"
