#!/usr/bin/env bash
#SBATCH --job-name=llmtd-state-depth
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
GATES=(current stable stable3 current stable stable3)
CONFIGS=(full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_default_1033760 full_train_default_1033760 full_train_default_1033760)
CATALOGS=(expanded_constraint_catalog_1057680 expanded_constraint_catalog_1057680 expanded_constraint_catalog_1057680 cross_dataset_catalog_default_1058707_1 cross_dataset_catalog_default_1058707_1 cross_dataset_catalog_default_1058707_1)
TRAJECTORIES=(expanded_catalog_trajectory_1057681 expanded_catalog_trajectory_1057681 expanded_catalog_trajectory_1057681 cross_dataset_trajectory_default_1058708_1 cross_dataset_trajectory_default_1058708_1 cross_dataset_trajectory_default_1058708_1)
CUTOFFS=(.306 .306 .306 .245 .245 .245)

i=$SLURM_ARRAY_TASK_ID; dataset=${DATASETS[$i]}; gate=${GATES[$i]}; run=${CONFIGS[$i]}
catalog=${CATALOGS[$i]}; trajectory=${TRAJECTORIES[$i]}; cutoff=${CUTOFFS[$i]}
cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/$catalog/constraint_catalog.json" \
  --schedule "$A/outputs/$trajectory/typed_readability_schedule.csv" \
  --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
  --samples 8192 --steps 50 --seed 20261030 --min-support 24 --drift .10 \
  --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --cooldown-steps 3 \
  --schedule-mode common --common-cutoff "$cutoff" --gate-mode current --topology-mode dynamic \
  --state-gate "$gate" --active-min-t .10 --active-max-t .75 \
  --output "$A/outputs/state_depth_${dataset}_${gate}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
