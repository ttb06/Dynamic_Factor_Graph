#!/usr/bin/env bash
# Train News with the released TabDiff representation and discretization:
# regression target in the numerical block, 50 diffusion timesteps.
#SBATCH --job-name=llmtd-news-reg-official
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$A/outputs/full_train_news_official_regression_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/train_pilot.py" \
  --repo "$REPO" --dataset news \
  --base-config "$A/outputs/full_train_default_1033760/config.pkl" \
  --regression-target-mode numerical --num-timesteps 50 \
  --epochs 8000 --batch-size 4096 --learning-rate .001 \
  --save-every 2000 --seed 20261018 --output "$OUT"
