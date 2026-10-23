#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-public
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
: "${TRAIN_JOB_ID:?TRAIN_JOB_ID is required}"
train="$A/outputs/full_train_news_${TRAIN_JOB_ID}"
out="$A/outputs/news_public_constraints_${SLURM_JOB_ID}"
mkdir -p "$out/samples"
cd "$REPO"
"$PY" "$A/scripts/sample_tabdiff_checkpoint.py" --repo "$REPO" --dataset news \
  --config "$train/config.pkl" --checkpoint "$train/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261012 --output "$out/samples/tabdiff.csv"
"$PY" "$A/scripts/apply_constrained_dgm_news_layer.py" \
  --cdgm-repo "$ROOT/baseline_repos/constrained_dgm" --sample "$out/samples/tabdiff.csv" \
  --output "$out/samples/tabdiff_cdgm_constraint_layer.csv"
