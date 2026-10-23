#!/usr/bin/env bash
#SBATCH --job-name=llmtd-scale-val-common
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
seed=$((20261368 + i))
base=$((i * 3))
d2="$A/outputs/default_stable8_scale_validation_s2_seed${seed}_1062656_${base}/samples"
d3="$A/outputs/default_stable8_scale_validation_s3_seed${seed}_1062656_$((base + 1))/samples"
d4="$A/outputs/default_stable8_scale_validation_s4_seed${seed}_1062656_$((base + 2))/samples"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" --seed "$seed" \
  --sample "unconditional=$d4/unconditional.csv" \
  --sample "final=$d4/final_reward_calibrated.csv" \
  --sample "scale2=$d2/x0_reward_calibrated_then_final_reward.csv" \
  --sample "scale3=$d3/x0_reward_calibrated_then_final_reward.csv" \
  --sample "scale4=$d4/x0_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/default_scale_validation_common_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
