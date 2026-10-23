#!/usr/bin/env bash
#SBATCH --job-name=llmtd-diabetes-traj
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CATALOG_JOB_ID:?CATALOG_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/typed_graph_trajectory_probe.py" --repo "$REPO" --dataset diabetes \
  --graph "$A/outputs/diabetes_catalog_${CATALOG_JOB_ID}/constraint_catalog.json" \
  --config "$A/outputs/full_train_diabetes_1048479/config.pkl" \
  --checkpoint "$A/outputs/full_train_diabetes_1048479/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261025 \
  --output "$A/outputs/diabetes_trajectory_${SLURM_JOB_ID}"
