#!/usr/bin/env bash
#SBATCH --job-name=llmtd-ext-probe
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${DATASET:?}"; : "${GRAPH_JOB_ID:?}"; : "${TRAIN_JOB_ID:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/typed_graph_trajectory_probe.py" --repo "$REPO" --dataset "$DATASET" \
  --graph "$A/outputs/extended_bscg_${DATASET}_${GRAPH_JOB_ID}/graph.json" \
  --config "$A/outputs/extended_full_train_${DATASET}_${TRAIN_JOB_ID}/config.pkl" \
  --checkpoint "$A/outputs/extended_full_train_${DATASET}_${TRAIN_JOB_ID}/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261440 --precision .90 --recall .80 \
  --output "$A/outputs/extended_dynamic_trajectory_${DATASET}_${SLURM_JOB_ID}"
