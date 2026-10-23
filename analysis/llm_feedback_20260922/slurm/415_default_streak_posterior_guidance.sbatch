#!/usr/bin/env bash
#SBATCH --job-name=llmtd-streak-guidance
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

: "${CALIB_JOB_ID:?submit with --export=ALL,CALIB_JOB_ID=<calibration job id>}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
seed=$((20261376 + i / 2))
variant=$((i % 2))
profiles=(rank rank_mass)
profile=${profiles[$variant]}
extra=()
if [[ "$variant" -eq 1 ]]; then
  extra=(--trajectory-profile "$A/outputs/default_streak_calibration_${CALIB_JOB_ID}/edge_state_trajectory.csv")
fi
full_graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$full_graph" --final-graph "$full_graph" \
  --evaluation-graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" \
  --schedule "$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv" \
  --streak-profile "$A/outputs/default_streak_calibration_${CALIB_JOB_ID}/violation_streak_calibration.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 1 \
  --max-actions-per-row-edge 1 --cooldown-steps 3 --gate-ranking trajectory_posterior \
  --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode dynamic --state-gate annealed8 \
  --active-min-t .25 --active-max-t .50 --numeric-guidance-mode constant --numeric-guidance-scale 2 \
  --max-guidance-scale 16 --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 4 --guided-final-repair \
  "${extra[@]}" \
  --output "$A/outputs/default_streak_guidance_${profile}_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
