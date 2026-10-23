#!/usr/bin/env bash
#SBATCH --job-name=llmtd-traj-portfolio
#SBATCH --partition=cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${CATALOG_JOB_ID:?CATALOG_JOB_ID required}"
: "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
A=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers default diabetes); i=$SLURM_ARRAY_TASK_ID; dataset=${DATASETS[$i]}
"$PY" "$A/scripts/select_trajectory_actionable_portfolio.py" \
  --graph "$A/outputs/actionable_catalog_v2_${dataset}_${CATALOG_JOB_ID}_${i}/constraint_catalog.json" \
  --trajectory "$A/outputs/actionable_trajectory_v2_${dataset}_${TRAJECTORY_JOB_ID}_${i}/edge_state_trajectory.csv" \
  --schedule "$A/outputs/actionable_trajectory_v2_${dataset}_${TRAJECTORY_JOB_ID}_${i}/typed_readability_schedule.csv" \
  --max-actuators 8 --min-final-violations 2 --max-per-endpoint 1 \
  --output "$A/outputs/trajectory_portfolio_v2_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
