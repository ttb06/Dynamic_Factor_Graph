#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-llmguard
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${BUILD_JOB_ID:?BUILD_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
BUILD="$A/outputs/news_llm_guardrail_graph_${BUILD_JOB_ID}"
case "$SLURM_ARRAY_TASK_ID" in
  0) GRAPH="$BUILD/public_only_graph.json"; SCHEDULE="$BUILD/public_only_schedule.csv"; WEIGHT=1.0; LABEL=public_guard;;
  1) GRAPH="$BUILD/public_plus_llm_score_graph.json"; SCHEDULE="$BUILD/public_plus_llm_schedule.csv"; WEIGHT=0.5; LABEL=llmguard_w05;;
  2) GRAPH="$BUILD/public_plus_llm_score_graph.json"; SCHEDULE="$BUILD/public_plus_llm_schedule.csv"; WEIGHT=2.0; LABEL=llmguard_w20;;
esac
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" \
  --repo "$REPO" --dataset news --graph "$GRAPH" \
  --final-graph "$A/outputs/news_public_typed_graph_20261015/graph.json" \
  --evaluation-graph "$A/outputs/news_public_typed_graph_20261015/graph.json" \
  --schedule "$SCHEDULE" \
  --trajectory-profile "$A/outputs/news_public_dynamic_trajectory_1081001/edge_state_trajectory.csv" \
  --streak-profile "$A/outputs/news_public_dynamic_trajectory_1081001/violation_streak_calibration.csv" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261447 --max-actuation-calibration-rate .05 \
  --authority-mode graph_precursor --authority-min-surplus 0 \
  --online-modes x0_reward_calibrated --guided-final-repair \
  --schedule-mode edge --topology-mode row_ema_hysteresis --volatility-threshold .15 --volatility-ema .8 \
  --state-gate annealed8 --gate-mode persistent --preserve-satisfied-actuate \
  --numeric-guidance-mode hardness_adaptive --max-actions-per-row-edge 2 \
  --cooldown-steps 2 --action-hold-steps 2 --max-guidance-delta 3.0 \
  --density-weight 1.0 --edit-weight .5 --graph-weight "$WEIGHT" --edit-norm l2 \
  --balanced-column-projection --balanced-min-t .25 --balanced-routing trajectory_reliability \
  --gate-ranking trajectory_posterior --active-min-t .00 --active-max-t .65 \
  --output "$A/outputs/news_llm_guardrail_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${LABEL}"
