#!/usr/bin/env bash
#SBATCH --job-name=llmtd-expanded-diagnostic
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

: "${SCHEDULE_JOB_ID:?submit with --export=ALL,SCHEDULE_JOB_ID=<trajectory job id>}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
graph="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$graph" --final-graph "$graph" --evaluation-graph "$graph" \
  --schedule "$A/outputs/default_expanded_graph_trajectory_${SCHEDULE_JOB_ID}/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261376 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 1 \
  --max-actions-per-row-edge 1 --cooldown-steps 3 \
  --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode dynamic --state-gate annealed8 \
  --active-min-t .25 --active-max-t .50 --numeric-guidance-mode constant --numeric-guidance-scale 2 \
  --max-guidance-scale 16 --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 4 --guided-final-repair \
  --output "$A/outputs/default_expanded_graph_diagnostic_${SLURM_JOB_ID}"
