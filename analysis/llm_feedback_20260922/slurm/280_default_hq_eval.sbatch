#!/usr/bin/env bash
#SBATCH --job-name=llmtd-default-hq-eval
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CATALOG_JOB_ID:?CATALOG_JOB_ID required}"
: "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
: "${UNION_JOB_ID:?UNION_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
for version in old hq; do
  graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"
  schedule="$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv"
  if [[ "$version" == hq ]]; then
    graph="$A/outputs/default_hq_catalog_${CATALOG_JOB_ID}/constraint_catalog.json"
    schedule="$A/outputs/default_hq_trajectory_${TRAJECTORY_JOB_ID}/typed_readability_schedule.csv"
  fi
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
    --graph "$graph" --evaluation-graph "$A/outputs/default_hq_union_${UNION_JOB_ID}/evaluation_graph.json" \
    --schedule "$schedule" --config "$A/outputs/full_train_default_1033760/config.pkl" \
    --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
    --samples 8192 --steps 50 --seed 20261089 --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --graph-weight 1 --max-actions-per-row-edge 1 --cooldown-steps 3 \
    --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode dynamic \
    --state-gate stable5 --active-min-t .25 --active-max-t .50 --online-modes x0_reward_calibrated \
    --output "$A/outputs/default_hq_compare_${version}_${SLURM_JOB_ID}"
done
