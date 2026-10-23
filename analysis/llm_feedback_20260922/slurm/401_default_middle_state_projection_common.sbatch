#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mid-xstate-common
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

: "${PARENT_JOB_ID:?submit with --export=ALL,PARENT_JOB_ID=<array job id>}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
root="$A/outputs"
d0="$root/default_middle_state_projection_min25_${PARENT_JOB_ID}_0/samples"
d1="$root/default_middle_state_projection_min20_${PARENT_JOB_ID}_1/samples"
d2="$root/default_middle_state_projection_min15_${PARENT_JOB_ID}_2/samples"
d3="$root/default_middle_state_projection_min10_${PARENT_JOB_ID}_3/samples"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" --seed 20261376 \
  --sample "unconditional=$d0/unconditional.csv" \
  --sample "final=$d0/final_reward_calibrated.csv" \
  --sample "x0_mid25=$d0/x0_reward_calibrated_then_final_reward.csv" \
  --sample "xstate_mid25=$d0/xstate_reward_calibrated_then_final_reward.csv" \
  --sample "x0_mid20=$d1/x0_reward_calibrated_then_final_reward.csv" \
  --sample "xstate_mid20=$d1/xstate_reward_calibrated_then_final_reward.csv" \
  --sample "x0_mid15=$d2/x0_reward_calibrated_then_final_reward.csv" \
  --sample "xstate_mid15=$d2/xstate_reward_calibrated_then_final_reward.csv" \
  --sample "x0_mid10=$d3/x0_reward_calibrated_then_final_reward.csv" \
  --sample "xstate_mid10=$d3/xstate_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/default_middle_state_projection_common_${PARENT_JOB_ID}_${SLURM_JOB_ID}"
