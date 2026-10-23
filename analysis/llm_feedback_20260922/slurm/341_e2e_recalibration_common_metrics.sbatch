#!/usr/bin/env bash
#SBATCH --job-name=llmtd-e2e-recal-common
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${EVALUATION_JOB_ID:?EVALUATION_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
root="$A/outputs"
p0="$root/e2e_guidance_recal_pretrained_weak_a1_${EVALUATION_JOB_ID}_0/samples"
g0="$root/e2e_guidance_recal_graph5_weak_a1_${EVALUATION_JOB_ID}_6/samples"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" --seed 20261304 \
  --sample "Pretrained-Unconditional=$p0/unconditional.csv" \
  --sample "Pretrained-FinalRepair=$p0/final_reward_calibrated.csv" \
  --sample "Pretrained-weak-a1=$root/e2e_guidance_recal_pretrained_weak_a1_${EVALUATION_JOB_ID}_0/samples/x0_reward_calibrated.csv" \
  --sample "Pretrained-weak-a4=$root/e2e_guidance_recal_pretrained_weak_a4_${EVALUATION_JOB_ID}_1/samples/x0_reward_calibrated.csv" \
  --sample "Pretrained-moderate-a1=$root/e2e_guidance_recal_pretrained_moderate_a1_${EVALUATION_JOB_ID}_2/samples/x0_reward_calibrated.csv" \
  --sample "Pretrained-moderate-a4=$root/e2e_guidance_recal_pretrained_moderate_a4_${EVALUATION_JOB_ID}_3/samples/x0_reward_calibrated.csv" \
  --sample "Pretrained-base-a1=$root/e2e_guidance_recal_pretrained_base_a1_${EVALUATION_JOB_ID}_4/samples/x0_reward_calibrated.csv" \
  --sample "Pretrained-base-a4=$root/e2e_guidance_recal_pretrained_base_a4_${EVALUATION_JOB_ID}_5/samples/x0_reward_calibrated.csv" \
  --sample "GraphFT-Unconditional=$g0/unconditional.csv" \
  --sample "GraphFT-FinalRepair=$g0/final_reward_calibrated.csv" \
  --sample "GraphFT-weak-a1=$root/e2e_guidance_recal_graph5_weak_a1_${EVALUATION_JOB_ID}_6/samples/x0_reward_calibrated.csv" \
  --sample "GraphFT-weak-a4=$root/e2e_guidance_recal_graph5_weak_a4_${EVALUATION_JOB_ID}_7/samples/x0_reward_calibrated.csv" \
  --sample "GraphFT-moderate-a1=$root/e2e_guidance_recal_graph5_moderate_a1_${EVALUATION_JOB_ID}_8/samples/x0_reward_calibrated.csv" \
  --sample "GraphFT-moderate-a4=$root/e2e_guidance_recal_graph5_moderate_a4_${EVALUATION_JOB_ID}_9/samples/x0_reward_calibrated.csv" \
  --sample "GraphFT-base-a1=$root/e2e_guidance_recal_graph5_base_a1_${EVALUATION_JOB_ID}_10/samples/x0_reward_calibrated.csv" \
  --sample "GraphFT-base-a4=$root/e2e_guidance_recal_graph5_base_a4_${EVALUATION_JOB_ID}_11/samples/x0_reward_calibrated.csv" \
  --output "$A/outputs/e2e_recalibration_common_metrics_${SLURM_JOB_ID}"
