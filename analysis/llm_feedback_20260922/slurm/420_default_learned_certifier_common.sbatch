#!/usr/bin/env bash
#SBATCH --job-name=llmtd-learned-cert-common
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

: "${PARENT_JOB_ID:?submit with --export=ALL,PARENT_JOB_ID=<array job id>}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
seed=$((20261376 + i))
root="$A/outputs/default_learned_certifier_s${seed}_${PARENT_JOB_ID}_${i}"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" --seed "$seed" \
  --sample "unconditional=$root/samples/unconditional.csv" \
  --sample "final=$root/samples/final_reward_calibrated.csv" \
  --sample "learned_certifier=$root/samples/xstate_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/default_learned_certifier_common_s${seed}_${PARENT_JOB_ID}_${SLURM_ARRAY_JOB_ID}_${i}"
