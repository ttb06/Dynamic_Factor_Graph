#!/usr/bin/env bash
#SBATCH --job-name=llmtd-slack-common
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
point_dirs=(
  "$A/outputs/calibration_slack_point_1062535_0/samples"
  "$A/outputs/calibration_slack_s20261346_point_1062542_0/samples"
  "$A/outputs/calibration_slack_s20261347_point_1062542_3/samples"
)
slack_dirs=(
  "$A/outputs/calibration_slack_z05_1062535_2/samples"
  "$A/outputs/calibration_slack_s20261346_z05_1062542_1/samples"
  "$A/outputs/calibration_slack_s20261347_z05_1062542_4/samples"
)
seeds=(20261345 20261346 20261347)
i=$SLURM_ARRAY_TASK_ID

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" --seed "${seeds[$i]}" \
  --sample "unconditional=${point_dirs[$i]}/unconditional.csv" \
  --sample "point-final=${point_dirs[$i]}/final_reward_calibrated.csv" \
  --sample "point-middle=${point_dirs[$i]}/x0_reward_calibrated.csv" \
  --sample "slack-middle=${slack_dirs[$i]}/x0_reward_calibrated.csv" \
  --sample "slack-certified=${slack_dirs[$i]}/x0_reward_calibrated_then_final_reward.csv" \
  --sample "slack-final=${slack_dirs[$i]}/final_reward_calibrated.csv" \
  --output "$A/outputs/calibration_slack_common_s${seeds[$i]}_${SLURM_ARRAY_JOB_ID}_${i}"
