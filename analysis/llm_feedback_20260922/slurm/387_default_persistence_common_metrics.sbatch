#!/usr/bin/env bash
#SBATCH --job-name=llmtd-persist-common
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
seed=$((20261361 + i))
base=$((i * 3))
d5="$A/outputs/default_persistence_validation_stable5_s${seed}_1062636_${base}/samples"
d8="$A/outputs/default_persistence_validation_stable8_s${seed}_1062636_$((base + 1))/samples"
d12="$A/outputs/default_persistence_validation_stable12_s${seed}_1062636_$((base + 2))/samples"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" --seed "$seed" \
  --sample "unconditional=$d8/unconditional.csv" \
  --sample "final=$d8/final_reward_calibrated.csv" \
  --sample "stable5=$d5/x0_reward_calibrated_then_final_reward.csv" \
  --sample "stable8=$d8/x0_reward_calibrated_then_final_reward.csv" \
  --sample "stable12=$d12/x0_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/default_persistence_common_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
