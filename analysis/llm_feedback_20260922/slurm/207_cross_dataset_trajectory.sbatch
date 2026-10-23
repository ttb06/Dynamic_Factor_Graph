#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xdata-traj
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${CATALOG_JOB_ID:?CATALOG_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(adult default)
CONFIGS=(full_train_1028011 full_train_default_1033760)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
run=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
graph="$A/outputs/cross_dataset_catalog_${dataset}_${CATALOG_JOB_ID}_${SLURM_ARRAY_TASK_ID}/constraint_catalog.json"
output="$A/outputs/cross_dataset_trajectory_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/typed_graph_trajectory_probe.py" \
  --repo "$REPO" --dataset "$dataset" --graph "$graph" \
  --config "$A/outputs/$run/config.pkl" \
  --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
  --samples 8192 --steps 50 --seed 20261007 --output "$output"
