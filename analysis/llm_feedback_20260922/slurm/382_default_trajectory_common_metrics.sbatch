#!/usr/bin/env bash
#SBATCH --job-name=llmtd-default-common
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
i=$SLURM_ARRAY_TASK_ID
seed=$((20261357 + i))
full_dirs=(
  "$A/outputs/default_trajectory_v2_full_s20261357_1062607_0/samples"
  "$A/outputs/default_trajectory_v2_full_s20261358_1062607_2/samples"
  "$A/outputs/default_trajectory_v2_full_s20261359_1062607_4/samples"
)
portfolio_dirs=(
  "$A/outputs/default_trajectory_v2_benefit_v2_s20261357_1062607_1/samples"
  "$A/outputs/default_trajectory_v2_benefit_v2_s20261358_1062607_3/samples"
  "$A/outputs/default_trajectory_v2_benefit_v2_s20261359_1062607_5/samples"
)

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" --seed "$seed" \
  --sample "unconditional=${full_dirs[$i]}/unconditional.csv" \
  --sample "final=${full_dirs[$i]}/final_reward_calibrated.csv" \
  --sample "full-middle=${full_dirs[$i]}/x0_reward_calibrated.csv" \
  --sample "portfolio-middle=${portfolio_dirs[$i]}/x0_reward_calibrated.csv" \
  --sample "portfolio-certified=${portfolio_dirs[$i]}/x0_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/default_trajectory_common_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
