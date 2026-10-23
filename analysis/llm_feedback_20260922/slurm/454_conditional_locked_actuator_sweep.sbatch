#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cond-actuator
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-7
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
i=$SLURM_ARRAY_TASK_ID
requests=(range range range category category both both both)
margins=(.1 .1 .1 0 0 .1 .1 .1)
scales=(2 4 4 2 2 2 4 4)
holds=(0 0 2 2 4 0 0 2)
request=${requests[$i]}
margin=${margins[$i]}
scale=${scales[$i]}
hold=${holds[$i]}
graph="$A/outputs/harpoon_conditional_graphs_${GRAPH_JOB_ID}/${request}.json"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$graph" --final-graph "$graph" --evaluation-graph "$graph" \
  --schedule "$A/outputs/harpoon_conditional_graphs_${GRAPH_JOB_ID}/schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261376 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 0 --action-margin-sd "$margin" \
  --max-actions-per-row-edge 2 --cooldown-steps 3 \
  --schedule-mode common --common-cutoff .50 --gate-mode current \
  --topology-mode dynamic --state-gate annealed8 \
  --active-min-t .25 --active-max-t .50 \
  --numeric-guidance-mode constant --numeric-guidance-scale "$scale" \
  --max-guidance-scale 16 --action-hold-steps "$hold" \
  --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 4 --lock-applicability-columns \
  --guided-final-repair \
  --output "$A/outputs/conditional_actuator_${request}_m${margin}_s${scale}_h${hold}_${SLURM_ARRAY_JOB_ID}_${i}"
