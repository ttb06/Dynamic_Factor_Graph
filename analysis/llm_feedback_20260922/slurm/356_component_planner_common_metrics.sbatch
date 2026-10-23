#!/usr/bin/env bash
#SBATCH --job-name=llmtd-component-common
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
: "${PARENT_JOB_ID:?PARENT_JOB_ID is required}"
root="$A/outputs"
s1="$root/component_planner_full_s1_g1_${PARENT_JOB_ID}_0/samples"
s2="$root/component_planner_full_s2_g1_${PARENT_JOB_ID}_1/samples"
s4="$root/component_planner_full_s4_g1_${PARENT_JOB_ID}_2/samples"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" --seed 20261321 \
  --sample "unconditional=$s1/unconditional.csv" \
  --sample "final=$s1/final_reward_calibrated.csv" \
  --sample "component-s1=$s1/x0_reward_calibrated.csv" \
  --sample "component-s1-certified=$s1/x0_reward_calibrated_then_final_reward.csv" \
  --sample "component-s2=$s2/x0_reward_calibrated.csv" \
  --sample "component-s2-certified=$s2/x0_reward_calibrated_then_final_reward.csv" \
  --sample "component-s4=$s4/x0_reward_calibrated.csv" \
  --output "$A/outputs/component_planner_common_${SLURM_JOB_ID}"
