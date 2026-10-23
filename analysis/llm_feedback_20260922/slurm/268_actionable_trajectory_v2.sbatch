#!/usr/bin/env bash
#SBATCH --job-name=llmtd-traj-v2
#SBATCH --partition=prioritized
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${CATALOG_JOB_ID:?CATALOG_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers default diabetes)
RUNS=(full_train_shoppers_1033761 full_train_default_1033760 full_train_diabetes_1048479)
SAMPLES=(8192 8192 4096)
i=$SLURM_ARRAY_TASK_ID; dataset=${DATASETS[$i]}; run=${RUNS[$i]}; samples=${SAMPLES[$i]}
"$PY" "$A/scripts/typed_graph_trajectory_probe.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/actionable_catalog_v2_${dataset}_${CATALOG_JOB_ID}_${i}/constraint_catalog.json" \
  --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
  --samples "$samples" --steps 50 --seed 20261085 \
  --output "$A/outputs/actionable_trajectory_v2_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
