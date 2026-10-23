#!/usr/bin/env bash
#SBATCH --job-name=llmtd-combined-confirm
#SBATCH --partition=prioritized
#SBATCH --array=0-3
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
DATASETS=(default default diabetes diabetes)
SEEDS=(20261094 20261095 20261094 20261095)
i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}
seed=${SEEDS[$i]}
samples=8192

if [[ "$dataset" == default ]]; then
  graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  schedule="$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv"
  run=full_train_default_1033760
  state_gate=stable5
  variants=(c2 c4)
else
  graph="$A/outputs/trajectory_portfolio_v2_diabetes_1060979_2/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_diabetes_1060955_2/evaluation_graph.json"
  schedule="$A/outputs/actionable_trajectory_v2_diabetes_1060954_2/typed_readability_schedule.csv"
  run=full_train_diabetes_1048479
  state_gate=current
  samples=4096
  variants=(c1)
fi

for variant in "${variants[@]}"; do
  scale=${variant#c}
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
    --graph "$graph" --evaluation-graph "$evaluation" --schedule "$schedule" \
    --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
    --samples "$samples" --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --graph-weight 1 \
    --max-actions-per-row-edge 1 --cooldown-steps 3 --schedule-mode common --common-cutoff .50 \
    --gate-mode current --topology-mode dynamic --state-gate "$state_gate" \
    --active-min-t .25 --active-max-t .50 --numeric-guidance-mode constant --numeric-guidance-scale "$scale" \
    --guided-final-repair --online-modes x0_reward_calibrated \
    --output "$A/outputs/combined_confirm_${dataset}_${variant}_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
done

hash_count=$(sha256sum "$A"/outputs/combined_confirm_${dataset}_*_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}/samples/unconditional.csv | cut -d' ' -f1 | sort -u | wc -l)
test "$hash_count" -eq 1
