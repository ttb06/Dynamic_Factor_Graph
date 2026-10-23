#!/usr/bin/env bash
#SBATCH --job-name=llmtd-default-midgraph
#SBATCH --partition=prioritized
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${CATALOG_JOB_ID:?CATALOG_JOB_ID required}"
: "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
TOPOLOGIES=(dynamic frozen periodic shuffled)
INTERVALS=(1 1 4 1)
topology=${TOPOLOGIES[$SLURM_ARRAY_TASK_ID]}
interval=${INTERVALS[$SLURM_ARRAY_TASK_ID]}
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/cross_dataset_catalog_default_${CATALOG_JOB_ID}_1/constraint_catalog.json" \
  --schedule "$A/outputs/cross_dataset_trajectory_default_${TRAJECTORY_JOB_ID}_1/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261009 --min-support 12 --drift .10 \
  --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --cooldown-steps 3 \
  --schedule-mode edge --gate-mode current --topology-mode "$topology" \
  --topology-refresh-interval "$interval" --active-min-t .10 --active-max-t .75 \
  --output "$A/outputs/default_middle_topology_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${topology}"
