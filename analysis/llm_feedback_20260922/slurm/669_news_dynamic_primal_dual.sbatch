#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-primaldual
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$SLURM_ARRAY_TASK_ID" in
  0) LR=.5; THRESHOLD=1.0; LABEL=conservative;;
  1) LR=1.0; THRESHOLD=.5; LABEL=responsive;;
esac
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_public_typed_graph_20261015/graph.json" \
  --final-graph "$A/outputs/news_public_typed_graph_20261015/graph.json" \
  --evaluation-graph "$A/outputs/news_public_typed_graph_20261015/graph.json" \
  --schedule "$A/outputs/news_public_dynamic_trajectory_1081001/typed_readability_schedule.csv" \
  --trajectory-profile "$A/outputs/news_public_dynamic_trajectory_1081001/edge_state_trajectory.csv" \
  --streak-profile "$A/outputs/news_public_dynamic_trajectory_1081001/violation_streak_calibration.csv" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261447 --max-actuation-calibration-rate .05 \
  --authority-mode graph_precursor --authority-min-surplus 0 \
  --online-modes x0_reward_calibrated --guided-final-repair \
  --schedule-mode edge --topology-mode row_ema_hysteresis --volatility-threshold .15 --volatility-ema .8 \
  --state-gate dual --gate-mode current --numeric-guidance-mode primal_dual \
  --dual-lr "$LR" --dual-decay .9 --dual-max 8 --dual-threshold "$THRESHOLD" \
  --numeric-guidance-scale 1.0 --max-actions-per-row-edge 3 \
  --cooldown-steps 1 --action-hold-steps 1 --max-guidance-delta 3.0 \
  --density-weight 1.0 --edit-weight .5 --edit-norm l2 \
  --balanced-column-projection --balanced-min-t .25 --balanced-routing trajectory_reliability \
  --gate-ranking trajectory_posterior --active-min-t .00 --active-max-t .65 \
  --output "$A/outputs/news_primal_dual_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${LABEL}"
