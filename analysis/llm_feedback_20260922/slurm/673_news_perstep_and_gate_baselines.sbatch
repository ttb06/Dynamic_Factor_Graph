#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-stepbase
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
COMMON=(
  --repo "$REPO" --dataset news
  --graph "$A/outputs/news_public_typed_graph_20261015/graph.json"
  --final-graph "$A/outputs/news_public_typed_graph_20261015/graph.json"
  --evaluation-graph "$A/outputs/news_public_typed_graph_20261015/graph.json"
  --schedule "$A/outputs/news_public_dynamic_trajectory_1081001/typed_readability_schedule.csv"
  --trajectory-profile "$A/outputs/news_public_dynamic_trajectory_1081001/edge_state_trajectory.csv"
  --streak-profile "$A/outputs/news_public_dynamic_trajectory_1081001/violation_streak_calibration.csv"
  --config "$A/outputs/full_train_news_1069382/config.pkl"
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt"
  --samples 4096 --steps 50 --seed 20261447
  --max-actuation-calibration-rate .05 --authority-mode graph_precursor --authority-min-surplus 0
  --online-modes x0_reward_calibrated --guided-final-repair
  --density-weight 1.0 --edit-weight .5 --graph-weight 0 --edit-norm l2
  --balanced-column-projection --balanced-routing fixed --gate-ranking trajectory_posterior
)
case "$SLURM_ARRAY_TASK_ID" in
  0)
    LABEL=full_perstep_projection
    EXTRA=(--schedule-mode all --topology-mode dynamic --state-gate current --gate-mode current
           --active-min-t 0 --active-max-t 1 --balanced-min-t 0
           --numeric-guidance-mode constant --max-actions-per-row-edge 0
           --cooldown-steps 0 --action-hold-steps 0 --max-guidance-delta 0)
    ;;
  1)
    LABEL=middle_current_gate
    EXTRA=(--schedule-mode edge --topology-mode row_ema_hysteresis --volatility-threshold .15 --volatility-ema .8
           --state-gate current --gate-mode current --active-min-t 0 --active-max-t .65 --balanced-min-t .25
           --numeric-guidance-mode hardness_adaptive --max-actions-per-row-edge 2
           --cooldown-steps 2 --action-hold-steps 2 --max-guidance-delta 3)
    ;;
esac
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" "${COMMON[@]}" "${EXTRA[@]}" \
  --output "$A/outputs/news_step_baseline_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${LABEL}"
