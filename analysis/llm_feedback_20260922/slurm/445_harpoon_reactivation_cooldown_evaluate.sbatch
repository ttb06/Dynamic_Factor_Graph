#!/usr/bin/env bash
#SBATCH --job-name=llmtd-react-cool-eval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

: "${PARENT_JOB_ID:?PARENT_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
request_i=$((i / 2))
cooldown_i=$((i % 2))
requests=(range both)
cooldowns=(1 5)
request=${requests[$request_i]}
cooldown=${cooldowns[$cooldown_i]}
root="$A/outputs/harpoon_reactivate_${request}_cool${cooldown}_${PARENT_JOB_ID}_${i}/samples"

"$PY" "$A/scripts/evaluate_harpoon_constraint_suite.py" --repo "$REPO" --request "$request" \
  --sample "unconditional=$root/unconditional.csv" \
  --sample "final_repair=$root/final_reward_calibrated.csv" \
  --sample "reactivate_cool${cooldown}=$root/xstate_reward_calibrated.csv" \
  --sample "reactivate_cool${cooldown}_then_certifier=$root/xstate_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/harpoon_reactivate_cool_eval_${request}_c${cooldown}_${SLURM_ARRAY_JOB_ID}_${i}"
