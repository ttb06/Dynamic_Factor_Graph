#!/usr/bin/env bash
#SBATCH --job-name=llmtd-base-metrics
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CATALOG_JOB_ID:?}"; : "${LOOKAHEAD_JOB_ID:?}"; : "${ACTION_JOB_ID:?}"; : "${SDV_JOB_ID:?}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/cuts/bin/python
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --graph "$A/outputs/expanded_constraint_catalog_${CATALOG_JOB_ID}/constraint_catalog.json" \
  --sample "TabDiff=$A/outputs/expanded_catalog_lookahead_${LOOKAHEAD_JOB_ID}/samples/official_unconditional.csv" \
  --sample "DynamicGraph-MidLookahead=$A/outputs/expanded_catalog_lookahead_${LOOKAHEAD_JOB_ID}/samples/mid_lookahead_select.csv" \
  --sample "DynamicGraph-ReverseX0=$A/outputs/expanded_catalog_reverse_action_${ACTION_JOB_ID}/samples/x0_reward_calibrated.csv" \
  --sample "PostRepair=$A/outputs/expanded_catalog_lookahead_${LOOKAHEAD_JOB_ID}/samples/final_reward_calibrated.csv" \
  --sample "CTGAN-SDV=$A/outputs/sdv_shoppers_ctgan_${SDV_JOB_ID}/synthetic.csv" \
  --sample "TVAE-SDV=$A/outputs/sdv_shoppers_tvae_${SDV_JOB_ID}/synthetic.csv" \
  --output "$A/outputs/shoppers_baseline_metric_suite_${SLURM_JOB_ID}"
