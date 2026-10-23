#!/usr/bin/env bash
#SBATCH --job-name=llmtd-shop-common
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
seeds=(20261383 20261384 20261385)
dyn=(shoppers_promote_confirm_s20261383_dynamic_1067743_0 shoppers_promote_confirm_s20261384_dynamic_1067743_2 shoppers_promote_confirm_s20261385_dynamic_1067743_4)
fro=(shoppers_promote_confirm_s20261383_frozen_state_1067743_1 shoppers_promote_confirm_s20261384_frozen_state_1067743_3 shoppers_promote_confirm_s20261385_frozen_state_1067743_5)
seed=${seeds[$i]}
dynamic_run="$A/outputs/${dyn[$i]}"
frozen_run="$A/outputs/${fro[$i]}"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset shoppers --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --sample "unconditional=$dynamic_run/samples/unconditional.csv" \
  --sample "final_promoted=$dynamic_run/samples/final_promoted_reward_calibrated.csv" \
  --sample "frozen_promoted=$frozen_run/samples/x0_reward_calibrated_then_final_promoted_reward.csv" \
  --sample "dynamic_promoted=$dynamic_run/samples/x0_reward_calibrated_then_final_promoted_reward.csv" \
  --seed "$seed" --output "$A/outputs/shoppers_promoted_common_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
