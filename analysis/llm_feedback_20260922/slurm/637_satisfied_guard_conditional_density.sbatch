#!/usr/bin/env bash
#SBATCH --job-name=llmtd-satguard
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID
labels=(guard_w1 conditional_w1 guard_conditional_w1 guard_conditional_w2)
graph_weights=(1 0 1 1)
conditional_weights=(0 1 1 2)
label=${labels[$i]}
extra=()
if [[ "$i" != "1" ]]; then extra+=(--preserve-satisfied-actuate); fi

cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" \
  --repo "$REPO" --dataset default \
  --graph "$A/outputs/bootstrap_conditional_graph_default_1070443_0/graph.json" \
  --final-graph "$A/outputs/bootstrap_conditional_graph_default_1070443_0/graph.json" \
  --evaluation-graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" \
  --schedule "$A/outputs/bscg_dynamic_trajectory_default_1080086_0/typed_readability_schedule.csv" \
  --trajectory-profile "$A/outputs/bscg_dynamic_trajectory_default_1080086_0/edge_state_trajectory.csv" \
  --streak-profile "$A/outputs/bscg_dynamic_trajectory_default_1080086_0/violation_streak_calibration.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261447 --max-actuation-calibration-rate .05 \
  --authority-mode graph_precursor --authority-min-surplus 0 \
  --online-modes x0_reward_calibrated --guided-final-repair \
  --schedule-mode edge --topology-mode row_ema_hysteresis --volatility-threshold .15 --volatility-ema .8 \
  --state-gate annealed8 --gate-mode persistent \
  --numeric-guidance-mode hardness_adaptive --max-actions-per-row-edge 2 \
  --cooldown-steps 2 --action-hold-steps 2 --max-guidance-delta 3.0 \
  --density-weight 1.0 --edit-weight .5 --graph-weight "${graph_weights[$i]}" \
  --conditional-density-weight "${conditional_weights[$i]}" --conditional-neighbors 64 \
  --gate-ranking trajectory_posterior --active-min-t .00 --active-max-t .65 \
  "${extra[@]}" \
  --output "$A/outputs/satisfied_guard_density_${label}_${SLURM_ARRAY_JOB_ID}_${i}"
