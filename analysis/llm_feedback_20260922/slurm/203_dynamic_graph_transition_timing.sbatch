#!/usr/bin/env bash
#SBATCH --job-name=llmtd-time-transition
#SBATCH --partition=prioritized
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${CATALOG_JOB_ID:?}"; : "${TRAJECTORY_JOB_ID:?}"
TIMES=(0.28 0.22 0.18); TARGET_T="${TIMES[$SLURM_ARRAY_TASK_ID]}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/rowwise_lookahead_branch.py" --repo "$REPO" \
  --graph "$A/outputs/expanded_constraint_catalog_${CATALOG_JOB_ID}/constraint_catalog.json" \
  --schedule "$A/outputs/expanded_catalog_trajectory_${TRAJECTORY_JOB_ID}/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261008 --target-t "$TARGET_T" \
  --min-support 12 --drift .10 \
  --output "$A/outputs/dynamic_graph_transition_timing_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
