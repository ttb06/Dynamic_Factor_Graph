#!/usr/bin/env bash
#SBATCH --job-name=llmtd-promote-calibrate
#SBATCH --partition=prioritized
#SBATCH --array=0-1
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
DATASETS=(shoppers default)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
seed=20261115
if [[ "$dataset" == shoppers ]]; then
  graph="$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"
  schedule="$A/outputs/actionable_trajectory_v2_shoppers_1060954_0/typed_readability_schedule.csv"
  run=full_train_shoppers_1033761; state_gate=current
else
  graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  schedule="$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv"
  run=full_train_default_1033760; state_gate=stable5
fi

LABELS=(r005s3 r010s3 r015s3 r020s2 r020s3 r020s5 r030s3 r050s3)
RATES=(.005 .010 .015 .020 .020 .020 .030 .050)
STREAKS=(3 3 3 2 3 5 3 3)
for k in "${!LABELS[@]}"; do
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
    --graph "$graph" --evaluation-graph "$evaluation" --schedule "$schedule" \
    --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
    --samples 8192 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --graph-weight 1 --max-actions-per-row-edge 1 \
    --cooldown-steps 3 --schedule-mode common --common-cutoff .75 --gate-mode current \
    --topology-mode dynamic --state-gate "$state_gate" --active-min-t .15 --active-max-t .60 \
    --numeric-guidance-mode constant --numeric-guidance-scale 4 --max-guidance-delta 0 \
    --promote-score-only --promoted-min-streak "${STREAKS[$k]}" \
    --promoted-max-calibration-rate "${RATES[$k]}" --online-modes x0_reward_calibrated \
    --output "$A/outputs/promotion_calibration_${dataset}_${LABELS[$k]}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done

hash_count=$(sha256sum "$A"/outputs/promotion_calibration_${dataset}_*_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples/unconditional.csv | cut -d' ' -f1 | sort -u | wc -l)
test "$hash_count" -eq 1
