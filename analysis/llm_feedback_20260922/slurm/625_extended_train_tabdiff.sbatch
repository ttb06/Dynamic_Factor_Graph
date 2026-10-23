#!/usr/bin/env bash
#SBATCH --job-name=llmtd-ext-train
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${DATASET:?DATASET required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/train_pilot.py" --repo "$REPO" --dataset "$DATASET" \
  --base-config "$A/outputs/full_train_default_1033760/config.pkl" \
  --epochs 8000 --batch-size 4096 --learning-rate .001 --save-every 2000 --seed 20261015 \
  --output "$A/outputs/extended_full_train_${DATASET}_${SLURM_JOB_ID}"
