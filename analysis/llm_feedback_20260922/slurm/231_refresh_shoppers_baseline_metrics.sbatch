#!/usr/bin/env bash
#SBATCH --job-name=llmtd-refresh-base
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/cuts/bin/python
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/expanded_constraint_catalog_1057680/constraint_catalog.json" \
  --sample "TabDiff=$A/outputs/expanded_catalog_lookahead_1057682/samples/official_unconditional.csv" \
  --sample "DynamicGraph-MidLookahead=$A/outputs/expanded_catalog_lookahead_1057682/samples/mid_lookahead_select.csv" \
  --sample "DynamicGraph-ReverseX0=$A/outputs/expanded_catalog_reverse_action_1057683/samples/x0_reward_calibrated.csv" \
  --sample "PostRepair=$A/outputs/expanded_catalog_lookahead_1057682/samples/final_reward_calibrated.csv" \
  --sample "CTGAN-SDV=$A/outputs/sdv_shoppers_ctgan_1057684/synthetic.csv" \
  --sample "TVAE-SDV=$A/outputs/sdv_shoppers_tvae_1057684/synthetic.csv" \
  --output "$A/outputs/shoppers_baseline_with_alpha_beta_${SLURM_JOB_ID}"
