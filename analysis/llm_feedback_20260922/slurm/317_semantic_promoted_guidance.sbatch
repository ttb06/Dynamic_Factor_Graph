#!/usr/bin/env bash
#SBATCH --job-name=llmtd-semantic-guidance
#SBATCH --partition=prioritized
#SBATCH --array=0-2
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
DATASETS=(shoppers default diabetes)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}; samples=8192
GRAPH_PREFIX=${GRAPH_PREFIX:-semantic_promotion}
graph="$A/outputs/${GRAPH_PREFIX}_${dataset}_${PROMOTION_JOB_ID}_${SLURM_ARRAY_TASK_ID}/constraint_catalog.json"
if [[ "$dataset" == shoppers ]]; then
  evaluation="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"
  schedule="$A/outputs/actionable_trajectory_v2_shoppers_1060954_0/typed_readability_schedule.csv"
  run=full_train_shoppers_1033761; state_gate=current; scale=4
elif [[ "$dataset" == default ]]; then
  evaluation="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  schedule="$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv"
  run=full_train_default_1033760; state_gate=stable5; scale=4
else
  evaluation="$A/outputs/actionable_union_eval_diabetes_1060955_2/evaluation_graph.json"
  schedule="$A/outputs/actionable_trajectory_v2_diabetes_1060954_2/typed_readability_schedule.csv"
  run=full_train_diabetes_1048479; state_gate=current; scale=1; samples=4096
fi

for mode in constant feedback; do
  extra=(); numeric=constant; maxa=1; cooldown=3
  if [[ "$mode" == feedback ]]; then numeric=feedback_adaptive; maxa=4; cooldown=1; fi
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
    --graph "$graph" --evaluation-graph "$evaluation" --schedule "$schedule" \
    --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
    --samples "$samples" --steps 50 --seed 20261117 --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --graph-weight 1 --max-actions-per-row-edge "$maxa" \
    --cooldown-steps "$cooldown" --schedule-mode common --common-cutoff .75 --gate-mode current \
    --topology-mode dynamic --state-gate "$state_gate" --active-min-t .15 --active-max-t .60 \
    --numeric-guidance-mode "$numeric" --numeric-guidance-scale "$scale" \
    --feedback-repeat-growth 1.5 --feedback-late-multiplier 2 --max-guidance-scale 16 \
    --max-guidance-delta 0 --online-modes x0_reward_calibrated \
    --output "$A/outputs/semantic_promoted_guidance_${dataset}_${mode}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done
