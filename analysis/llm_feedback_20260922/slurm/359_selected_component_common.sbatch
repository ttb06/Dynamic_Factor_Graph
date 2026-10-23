#!/usr/bin/env bash
#SBATCH --job-name=llmtd-selected-common
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
: "${STRENGTH_JOB_ID:?STRENGTH_JOB_ID is required}"
: "${CERTIFIED_JOB_ID:?CERTIFIED_JOB_ID is required}"
selected="$A/outputs/component_strength_s150_${STRENGTH_JOB_ID}_1/samples"
filtered="$A/outputs/trajectory_certified_component_s2_g1_${CERTIFIED_JOB_ID}_1/samples"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" --seed 20261322 \
  --sample "unconditional=$selected/unconditional.csv" \
  --sample "final=$selected/final_reward_calibrated.csv" \
  --sample "component-s15=$selected/x0_reward_calibrated.csv" \
  --sample "component-s15-certified=$selected/x0_reward_calibrated_then_final_reward.csv" \
  --sample "filtered-s2=$filtered/x0_reward_calibrated.csv" \
  --sample "filtered-s2-certified=$filtered/x0_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/selected_component_common_${SLURM_JOB_ID}"
