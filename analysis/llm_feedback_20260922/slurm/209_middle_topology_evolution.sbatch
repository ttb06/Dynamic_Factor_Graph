#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mid-topology
#SBATCH --partition=prioritized
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
TOPOLOGIES=(dynamic frozen shuffled dynamic dynamic)
MIN_T=(.10 .10 .10 .18 .10)
MAX_T=(.75 .75 .75 .75 .18)
LABELS=(dynamic_full frozen_middle shuffled_rows middle_only late_middle_only)
topology=${TOPOLOGIES[$SLURM_ARRAY_TASK_ID]}
min_t=${MIN_T[$SLURM_ARRAY_TASK_ID]}
max_t=${MAX_T[$SLURM_ARRAY_TASK_ID]}
label=${LABELS[$SLURM_ARRAY_TASK_ID]}
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
  --topology-mode "$topology" --active-min-t "$min_t" --active-max-t "$max_t" \
  --output "$A/outputs/middle_topology_evolution_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${label}"
