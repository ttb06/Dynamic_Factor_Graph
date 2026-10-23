#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mid-xstate-confirm-common
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

: "${PARENT_JOB_ID:?submit with --export=ALL,PARENT_JOB_ID=<array job id>}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
seed=$((20261377 + i))
base=$((i * 2))
root="$A/outputs"
dynamic="$root/default_middle_state_confirm_dynamic_s${seed}_${PARENT_JOB_ID}_${base}/samples"
frozen="$root/default_middle_state_confirm_frozen_s${seed}_${PARENT_JOB_ID}_$((base + 1))/samples"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" --seed "$seed" \
  --sample "unconditional=$dynamic/unconditional.csv" \
  --sample "final=$dynamic/final_reward_calibrated.csv" \
  --sample "dynamic_x0=$dynamic/x0_reward_calibrated_then_final_reward.csv" \
  --sample "dynamic_xstate=$dynamic/xstate_reward_calibrated_then_final_reward.csv" \
  --sample "frozen_x0=$frozen/x0_reward_calibrated_then_final_reward.csv" \
  --sample "frozen_xstate=$frozen/xstate_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/default_middle_state_confirm_common_s${seed}_${PARENT_JOB_ID}_${SLURM_ARRAY_JOB_ID}_${i}"
