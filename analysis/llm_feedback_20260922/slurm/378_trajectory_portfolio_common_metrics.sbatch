#!/usr/bin/env bash
#SBATCH --job-name=llmtd-portfolio-common
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
seed=$((20261350 + i))
portfolio_dirs=(
  "$A/outputs/trajectory_benefit_benefit_portfolio_s20261350_1062570_1/samples"
  "$A/outputs/trajectory_benefit_benefit_portfolio_s20261351_1062570_3/samples"
  "$A/outputs/trajectory_benefit_benefit_portfolio_s20261352_1062570_5/samples"
)
full_dirs=(
  "$A/outputs/trajectory_benefit_full_s20261350_1062570_0/samples"
  "$A/outputs/trajectory_benefit_full_s20261351_1062570_2/samples"
  "$A/outputs/trajectory_benefit_full_s20261352_1062570_4/samples"
)

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" --seed "$seed" \
  --sample "unconditional=${portfolio_dirs[$i]}/unconditional.csv" \
  --sample "final=${portfolio_dirs[$i]}/final_reward_calibrated.csv" \
  --sample "portfolio-middle=${portfolio_dirs[$i]}/x0_reward_calibrated.csv" \
  --sample "portfolio-certified=${portfolio_dirs[$i]}/x0_reward_calibrated_then_final_reward.csv" \
  --sample "full-middle=${full_dirs[$i]}/x0_reward_calibrated.csv" \
  --sample "full-certified=${full_dirs[$i]}/x0_reward_calibrated_then_final_reward.csv" \
  --output "$A/outputs/trajectory_portfolio_common_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
