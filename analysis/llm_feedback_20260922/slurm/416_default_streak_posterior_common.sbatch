#!/usr/bin/env bash
#SBATCH --job-name=llmtd-streak-common
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5
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
seed=$((20261376 + i / 2))
variant=$((i % 2))
profiles=(rank rank_mass)
profile=${profiles[$variant]}
root="$A/outputs/default_streak_guidance_${profile}_s${seed}_${PARENT_JOB_ID}_${i}"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" --seed "$seed" \
  --sample "unconditional=$root/samples/unconditional.csv" \
  --sample "final=$root/samples/final_reward_calibrated.csv" \
  --sample "streak_posterior=$root/samples/xstate_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/default_streak_guidance_common_${profile}_s${seed}_${PARENT_JOB_ID}_${SLURM_ARRAY_JOB_ID}_${i}"
