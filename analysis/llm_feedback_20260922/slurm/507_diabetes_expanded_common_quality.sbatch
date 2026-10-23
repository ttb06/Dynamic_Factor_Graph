#!/usr/bin/env bash
#SBATCH --job-name=llmtd-diab-common
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
seeds=(20261383 20261384 20261385)
seed=${seeds[$i]}
run="$A/outputs/diabetes_expanded_confirm_s${seed}_dynamic_1068457_${i}"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset diabetes --graph "$A/outputs/actionable_union_eval_diabetes_1060955_2/evaluation_graph.json" \
  --sample "unconditional=$run/samples/unconditional.csv" \
  --sample "final_promoted=$run/samples/final_promoted_reward_calibrated.csv" \
  --sample "trajectory_only=$run/samples/x0_reward_calibrated.csv" \
  --sample "dynamic_full=$run/samples/x0_reward_calibrated_then_final_promoted_reward.csv" \
  --seed "$seed" --output "$A/outputs/diabetes_expanded_common_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
