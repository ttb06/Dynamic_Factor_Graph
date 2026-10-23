#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-hold
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
request_i=$((i / 2))
hold_i=$((i % 2))
requests=(range both)
schedule_indices=(0 2)
holds=(4 8)
request=${requests[$request_i]}
schedule_i=${schedule_indices[$request_i]}
hold=${holds[$hold_i]}
graph="$A/outputs/harpoon_default_graphs_1065641/${request}.json"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$graph" --final-graph "$graph" --evaluation-graph "$graph" \
  --schedule "$A/outputs/harpoon_default_trajectory_${request}_1065642_${schedule_i}/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261376 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 0 \
  --max-actions-per-row-edge 1 --cooldown-steps 3 --action-hold-steps "$hold" \
  --schedule-mode common --common-cutoff .50 --gate-mode current \
  --topology-mode dynamic --state-gate annealed8 \
  --active-min-t .25 --active-max-t .50 \
  --numeric-guidance-mode constant --numeric-guidance-scale 2 --max-guidance-scale 16 \
  --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 4 --guided-final-repair \
  --output "$A/outputs/harpoon_hold_${request}_h${hold}_${SLURM_ARRAY_JOB_ID}_${i}"
