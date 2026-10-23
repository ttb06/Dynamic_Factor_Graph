#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mid-refresh
#SBATCH --partition=prioritized
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
INTERVALS=(2 4)
interval=${INTERVALS[$SLURM_ARRAY_TASK_ID]}
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" \
  --graph "$A/outputs/expanded_constraint_catalog_1057680/constraint_catalog.json" \
  --schedule "$A/outputs/expanded_catalog_trajectory_1057681/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261009 --min-support 12 --drift .10 \
  --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --cooldown-steps 3 \
  --schedule-mode common --common-cutoff .306 --gate-mode current \
  --topology-mode periodic --topology-refresh-interval "$interval" --active-min-t .10 --active-max-t .75 \
  --output "$A/outputs/middle_topology_refresh_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_k${interval}"
