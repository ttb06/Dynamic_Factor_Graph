#!/usr/bin/env bash
#SBATCH --job-name=llmtd-main-quality
#SBATCH --partition=prioritized
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${MAIN_JOB_ID:?MAIN_JOB_ID required}"

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/cuts/bin/python
DATASETS=(shoppers shoppers shoppers default default default)
SEEDS=(20261021 20261022 20261023 20261021 20261022 20261023)
CATALOGS=(expanded_constraint_catalog_1057680 expanded_constraint_catalog_1057680 expanded_constraint_catalog_1057680 cross_dataset_catalog_default_1058707_1 cross_dataset_catalog_default_1058707_1 cross_dataset_catalog_default_1058707_1)

i=$SLURM_ARRAY_TASK_ID; dataset=${DATASETS[$i]}; seed=${SEEDS[$i]}; catalog=${CATALOGS[$i]}
prefix="$A/outputs/main_state_${dataset}_s${seed}"
current="${prefix}_dynamic_current_${MAIN_JOB_ID}_${i}"
stable="${prefix}_dynamic_stable_${MAIN_JOB_ID}_${i}"
frozen="${prefix}_frozen_state_${MAIN_JOB_ID}_${i}"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/$catalog/constraint_catalog.json" \
  --sample "TabDiff=$stable/samples/unconditional.csv" \
  --sample "Dynamic-current=$current/samples/x0_reward_calibrated.csv" \
  --sample "Dynamic-stable=$stable/samples/x0_reward_calibrated.csv" \
  --sample "Frozen-state=$frozen/samples/x0_reward_calibrated.csv" \
  --sample "Final-repair=$stable/samples/final_directed_calibrated.csv" \
  --output "$A/outputs/main_quality_${dataset}_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
