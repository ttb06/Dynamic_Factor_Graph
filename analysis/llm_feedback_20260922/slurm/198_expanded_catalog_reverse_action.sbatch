#!/usr/bin/env bash
#SBATCH --job-name=llmtd-expand-action
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CATALOG_JOB_ID:?CATALOG_JOB_ID required}"; : "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
: "${SAMPLES:=4096}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" \
  --graph "$A/outputs/expanded_constraint_catalog_${CATALOG_JOB_ID}/constraint_catalog.json" \
  --schedule "$A/outputs/expanded_catalog_trajectory_${TRAJECTORY_JOB_ID}/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples "$SAMPLES" --steps 50 --seed 20261007 --min-support 12 --drift .10 \
  --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --cooldown-steps 3 \
  --output "$A/outputs/expanded_catalog_reverse_action_${SLURM_JOB_ID}"
