#!/usr/bin/env bash
# Same-sample comparison with final repair and the optional certifier.
#SBATCH --job-name=llmtd-news-auto-matched
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SRC="$A/outputs/news_auto_knee_1093489_1_s100"
OUT="$A/outputs/news_auto_matched_eval_${SLURM_JOB_ID}"

cd "$REPO"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$SRC/samples/unconditional.csv" \
  --sample "final_repair=$SRC/samples/final_repair.csv" \
  --sample "auto_knee=$SRC/samples/dense_factor.csv" \
  --sample "auto_knee_then_certifier=$SRC/samples/dense_factor_then_certifier.csv" \
  --seed 20261607 --output "$OUT"
