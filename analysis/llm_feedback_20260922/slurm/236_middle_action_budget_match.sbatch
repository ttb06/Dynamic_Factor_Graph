#!/usr/bin/env bash
#SBATCH --job-name=llmtd-budget-match
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
SEEDS=(20261031 20261032 20261033 20261031 20261032 20261033)
REF_TASKS=(0 1 2 6 7 8)
i=$SLURM_ARRAY_TASK_ID; dataset=${DATASETS[$i]}; seed=${SEEDS[$i]}; ref_task=${REF_TASKS[$i]}

if [[ "$dataset" == shoppers ]]; then
  run=full_train_shoppers_1033761; catalog=expanded_constraint_catalog_1057680; trajectory=expanded_catalog_trajectory_1057681
else
  run=full_train_default_1033760; catalog=cross_dataset_catalog_default_1058707_1; trajectory=cross_dataset_trajectory_default_1058708_1
fi
reference="$A/outputs/time_seed_${dataset}_middle_s${seed}_frozen_state_1060317_${ref_task}/method_metrics.csv"
budget=$("$PY" -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); print(int(d.loc[d.method.eq("x0_reward_calibrated"),"action_events"].iloc[0]))' "$reference")
if [[ "$budget" -le 0 ]]; then echo "Invalid matched action budget: $budget" >&2; exit 2; fi

cd "$REPO"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/$catalog/constraint_catalog.json" \
  --schedule "$A/outputs/$trajectory/typed_readability_schedule.csv" \
  --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
  --samples 8192 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
  --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --max-total-actions "$budget" --cooldown-steps 3 \
  --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode dynamic \
  --state-gate stable --active-min-t .25 --active-max-t .50 \
  --output "$A/outputs/budget_match_${dataset}_middle_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
