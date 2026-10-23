#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cond-late-only
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

: "${GRAPH_JOB_ID:?GRAPH_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
requests=(range category both)
request=${requests[$SLURM_ARRAY_TASK_ID]}
margin=0
if [[ "$request" != category ]]; then margin=.1; fi
graph="$A/outputs/harpoon_conditional_graphs_${GRAPH_JOB_ID}/${request}.json"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$graph" --final-graph "$graph" --evaluation-graph "$graph" \
  --schedule "$A/outputs/harpoon_conditional_graphs_${GRAPH_JOB_ID}/schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261376 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 0 --action-margin-sd "$margin" \
  --max-actions-per-row-edge 1 --cooldown-steps 0 \
  --schedule-mode common --common-cutoff .50 --gate-mode current \
  --topology-mode dynamic --state-gate current \
  --active-min-t .09 --active-max-t .11 \
  --numeric-guidance-mode constant --numeric-guidance-scale 2 --max-guidance-scale 16 \
  --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 4 --lock-applicability-columns \
  --guided-final-repair \
  --output "$A/outputs/conditional_late_only_${request}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
