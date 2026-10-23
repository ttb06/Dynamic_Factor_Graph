#!/usr/bin/env bash
#SBATCH --job-name=llmtd-anneal-common
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
d0="$root/default_annealed_break_stable5_${PARENT_JOB_ID}_0/samples"
d1="$root/default_annealed_break_stable8_${PARENT_JOB_ID}_1/samples"
d2="$root/default_annealed_break_annealed8_${PARENT_JOB_ID}_2/samples"
d3="$root/default_annealed_break_annealed12_${PARENT_JOB_ID}_3/samples"
d4="$root/default_annealed_break_annealed8_step10_${PARENT_JOB_ID}_4/samples"
d5="$root/default_annealed_break_annealed8_step25_${PARENT_JOB_ID}_5/samples"
d6="$root/default_annealed_break_annealed8_feedback2_${PARENT_JOB_ID}_6/samples"
d7="$root/default_annealed_break_annealed8_feedback4_${PARENT_JOB_ID}_7/samples"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" --seed 20261371 \
  --sample "unconditional=$d0/unconditional.csv" \
  --sample "final=$d0/final_reward_calibrated.csv" \
  --sample "stable5=$d0/x0_reward_calibrated_then_final_reward.csv" \
  --sample "stable8=$d1/x0_reward_calibrated_then_final_reward.csv" \
  --sample "annealed8=$d2/x0_reward_calibrated_then_final_reward.csv" \
  --sample "annealed12=$d3/x0_reward_calibrated_then_final_reward.csv" \
  --sample "annealed8_step10=$d4/x0_reward_calibrated_then_final_reward.csv" \
  --sample "annealed8_step25=$d5/x0_reward_calibrated_then_final_reward.csv" \
  --sample "annealed8_feedback2=$d6/x0_reward_calibrated_then_final_reward.csv" \
  --sample "annealed8_feedback4=$d7/x0_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/default_annealed_break_common_${PARENT_JOB_ID}_${SLURM_JOB_ID}"
