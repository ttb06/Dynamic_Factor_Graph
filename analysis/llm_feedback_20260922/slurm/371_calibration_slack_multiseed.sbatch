#!/usr/bin/env bash
#SBATCH --job-name=llmtd-slack-seeds
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

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
seeds=(20261346 20261346 20261346 20261347 20261347 20261347)
zs=(0 .5 1 0 .5 1)
profiles=(point z05 z1 point z05 z1)
i=$SLURM_ARRAY_TASK_ID
extra=(--calibration-policy point)
if [[ "${zs[$i]}" != 0 ]]; then extra=(--calibration-policy wilson_upper --calibration-z "${zs[$i]}"); fi

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" \
  --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json" \
  --evaluation-graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --schedule "$A/outputs/e2e_graph_trajectory_pretrained_1062231_0/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed "${seeds[$i]}" --min-support 12 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 1 --max-actions-per-row-edge 4 --cooldown-steps 1 \
  --schedule-mode edge --gate-mode current --topology-mode dynamic --state-gate current \
  --active-min-t .15 --active-max-t .60 --numeric-guidance-mode feedback_adaptive \
  --numeric-guidance-scale 1.5 --feedback-repeat-growth 1.5 --feedback-late-multiplier 2 \
  --max-guidance-scale 8 --online-modes x0_reward_calibrated \
  --action-planner component_beam --component-beam-width 4 --component-violation-weight 100 \
  --guided-final-repair "${extra[@]}" \
  --output "$A/outputs/calibration_slack_s${seeds[$i]}_${profiles[$i]}_${SLURM_ARRAY_JOB_ID}_${i}"
