#!/usr/bin/env bash
#SBATCH --job-name=llmtd-temp-gates
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
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

variants=(event_triggered progressive_gate shuffled_topology)
state_gates=(transition annealed8 stable)
topologies=(row_ema_hysteresis row_ema_hysteresis shuffled)
gate_modes=(current persistent persistent)
max_total_actions=(0 0 94)
i=$SLURM_ARRAY_TASK_ID
variant=${variants[$i]}
state_gate=${state_gates[$i]}
topology=${topologies[$i]}
gate_mode=${gate_modes[$i]}
max_total=${max_total_actions[$i]}

probe="$A/outputs/bscg_dynamic_trajectory_default_1080086_0"
graph="$A/outputs/bootstrap_conditional_graph_default_1070443_0/graph.json"
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" \
  --repo "$REPO" --dataset default --graph "$graph" --final-graph "$graph" \
  --evaluation-graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" \
  --schedule "$probe/typed_readability_schedule.csv" --trajectory-profile "$probe/edge_state_trajectory.csv" \
  --streak-profile "$probe/violation_streak_calibration.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261447 --max-actuation-calibration-rate .05 \
  --online-modes x0_reward_calibrated --guided-final-repair \
  --schedule-mode edge --topology-mode "$topology" --volatility-threshold .15 --volatility-ema .8 \
  --state-gate "$state_gate" --gate-mode "$gate_mode" \
  --numeric-guidance-mode hardness_adaptive --max-actions-per-row-edge 2 \
  --max-total-actions "$max_total" \
  --cooldown-steps 2 --action-hold-steps 2 --max-guidance-delta 3.0 \
  --density-weight 1.0 --edit-weight .5 --gate-ranking trajectory_posterior \
  --active-min-t .00 --active-max-t .65 \
  --output "$A/outputs/temporal_gate_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
