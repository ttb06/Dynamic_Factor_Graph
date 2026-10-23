#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixgraph-eval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

: "${FINETUNE_JOB_ID:?FINETUNE_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
VARIANTS=(head_continue head_calibrated full_continue full_calibrated)
i=$SLURM_ARRAY_TASK_ID
variant=${VARIANTS[$i]}
run="$A/outputs/mixed_graph_finetune_rerun_${variant}_${FINETUNE_JOB_ID}_${i}"
graph_dir="$A/outputs/natural_dag_stable_v3_20261010"
test -s "$run/ema_model_epoch_150.pt"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$graph_dir/graph.json" --final-graph "$graph_dir/graph.json" \
  --evaluation-graph "$graph_dir/graph.json" --schedule "$graph_dir/schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$run/ema_model_epoch_150.pt" \
  --samples 4096 --steps 50 --seed 20261376 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 0 \
  --max-actions-per-row-edge 3 --cooldown-steps 3 \
  --schedule-mode common --common-cutoff .50 --gate-mode current \
  --topology-mode dynamic --state-gate annealed8 --active-min-t .10 --active-max-t .50 \
  --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 8 --lock-applicability-columns \
  --calibration-policy wilson_upper --calibration-z 1 --guided-final-repair \
  --output "$A/outputs/mixed_graph_eval_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
