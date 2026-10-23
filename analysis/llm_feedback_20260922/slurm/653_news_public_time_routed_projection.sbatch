#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-route
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
GRAPH="$A/outputs/news_public_typed_graph_20261015/graph.json"
PROBE="$A/outputs/news_public_dynamic_trajectory_1081001"
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" \
  --repo "$REPO" --dataset news --graph "$GRAPH" --final-graph "$GRAPH" --evaluation-graph "$GRAPH" \
  --schedule "$PROBE/typed_readability_schedule.csv" --trajectory-profile "$PROBE/edge_state_trajectory.csv" \
  --streak-profile "$PROBE/violation_streak_calibration.csv" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261447 --max-actuation-calibration-rate .05 \
  --authority-mode graph_precursor --authority-min-surplus 0 \
  --online-modes x0_reward_calibrated --guided-final-repair \
  --schedule-mode edge --topology-mode row_ema_hysteresis --volatility-threshold .15 --volatility-ema .8 \
  --state-gate annealed8 --gate-mode persistent \
  --numeric-guidance-mode hardness_adaptive --max-actions-per-row-edge 2 \
  --cooldown-steps 2 --action-hold-steps 2 --max-guidance-delta 3.0 \
  --density-weight 1.0 --edit-weight .5 --edit-norm l2 \
  --balanced-column-projection --balanced-min-t .15 \
  --gate-ranking trajectory_posterior --active-min-t .00 --active-max-t .65 \
  --output "$A/outputs/news_projection_time_routed_${SLURM_JOB_ID}"
