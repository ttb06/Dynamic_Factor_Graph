#!/usr/bin/env bash
#SBATCH --job-name=llmtd-stable-window
#SBATCH --partition=prioritized
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers shoppers shoppers default default default)
CONFIGS=(full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_default_1033760 full_train_default_1033760 full_train_default_1033760)
CATALOGS=(expanded_constraint_catalog_1057680 expanded_constraint_catalog_1057680 expanded_constraint_catalog_1057680 cross_dataset_catalog_default_1058707_1 cross_dataset_catalog_default_1058707_1 cross_dataset_catalog_default_1058707_1)
TRAJECTORIES=(expanded_catalog_trajectory_1057681 expanded_catalog_trajectory_1057681 expanded_catalog_trajectory_1057681 cross_dataset_trajectory_default_1058708_1 cross_dataset_trajectory_default_1058708_1 cross_dataset_trajectory_default_1058708_1)
LABELS=(early middle late early middle late)
MINS=(.55 .25 .10 .55 .25 .10)
MAXS=(.75 .50 .25 .75 .50 .25)
CUTOFFS=(.75 .50 .25 .75 .50 .25)

i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}; run=${CONFIGS[$i]}; catalog=${CATALOGS[$i]}; trajectory=${TRAJECTORIES[$i]}
label=${LABELS[$i]}; min_t=${MINS[$i]}; max_t=${MAXS[$i]}; cutoff=${CUTOFFS[$i]}
cd "$REPO"
for topology in dynamic frozen_state; do
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
    --graph "$A/outputs/$catalog/constraint_catalog.json" \
    --schedule "$A/outputs/$trajectory/typed_readability_schedule.csv" \
    --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
    --samples 8192 --steps 50 --seed 20261024 --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --cooldown-steps 3 \
    --schedule-mode common --common-cutoff "$cutoff" --gate-mode current --topology-mode "$topology" \
    --state-gate stable --active-min-t "$min_t" --active-max-t "$max_t" \
    --output "$A/outputs/stable_window_${dataset}_${label}_${topology}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done
