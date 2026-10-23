#!/usr/bin/env bash
#SBATCH --job-name=llmtd-scale-common
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" --seed 20261367 \
  --sample "unconditional=$A/outputs/default_stable8_scale4_1062651_3/samples/unconditional.csv" \
  --sample "final=$A/outputs/default_stable8_scale4_1062651_3/samples/final_reward_calibrated.csv" \
  --sample "scale1=$A/outputs/default_stable8_scale1_1062651_0/samples/x0_reward_calibrated_then_final_reward.csv" \
  --sample "scale2=$A/outputs/default_stable8_scale2_1062651_1/samples/x0_reward_calibrated_then_final_reward.csv" \
  --sample "scale3=$A/outputs/default_stable8_scale3_1062651_2/samples/x0_reward_calibrated_then_final_reward.csv" \
  --sample "scale4=$A/outputs/default_stable8_scale4_1062651_3/samples/x0_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/default_stable8_scale_common_${SLURM_JOB_ID}"
