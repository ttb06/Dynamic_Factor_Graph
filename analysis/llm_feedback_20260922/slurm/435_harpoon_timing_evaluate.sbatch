#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-timing-eval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-8
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${PARENT_JOB_ID:?PARENT_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"; PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID; request_i=$((i / 3)); window_i=$((i % 3))
requests=(range category both); labels=(early middle late)
request=${requests[$request_i]}; label=${labels[$window_i]}
root="$A/outputs/harpoon_timing_${request}_${label}_${PARENT_JOB_ID}_${i}/samples"
"$PY" "$A/scripts/evaluate_harpoon_constraint_suite.py" --repo "$REPO" --request "$request" --seed 20261383 \
  --sample "unconditional=$root/unconditional.csv" \
  --sample "final_repair=$root/final_reward_calibrated.csv" \
  --sample "${label}_dynamic=$root/xstate_reward_calibrated.csv" \
  --sample "${label}_then_certifier=$root/xstate_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/harpoon_timing_eval_${request}_${label}_${SLURM_ARRAY_JOB_ID}_${i}"
