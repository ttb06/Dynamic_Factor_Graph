#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mid-anneal-common
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
d0="$root/default_middle_annealed_min25_s2_${PARENT_JOB_ID}_0/samples"
d1="$root/default_middle_annealed_min25_s4_${PARENT_JOB_ID}_1/samples"
d2="$root/default_middle_annealed_min20_s2_${PARENT_JOB_ID}_2/samples"
d3="$root/default_middle_annealed_min20_s4_${PARENT_JOB_ID}_3/samples"
d4="$root/default_middle_annealed_min15_s2_${PARENT_JOB_ID}_4/samples"
d5="$root/default_middle_annealed_min15_s4_${PARENT_JOB_ID}_5/samples"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" --seed 20261375 \
  --sample "unconditional=$d0/unconditional.csv" \
  --sample "final=$d0/final_reward_calibrated.csv" \
  --sample "mid25_scale2=$d0/x0_reward_calibrated_then_final_reward.csv" \
  --sample "mid25_scale4=$d1/x0_reward_calibrated_then_final_reward.csv" \
  --sample "mid20_scale2=$d2/x0_reward_calibrated_then_final_reward.csv" \
  --sample "mid20_scale4=$d3/x0_reward_calibrated_then_final_reward.csv" \
  --sample "mid15_scale2=$d4/x0_reward_calibrated_then_final_reward.csv" \
  --sample "mid15_scale4=$d5/x0_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/default_middle_annealed_common_${PARENT_JOB_ID}_${SLURM_JOB_ID}"
