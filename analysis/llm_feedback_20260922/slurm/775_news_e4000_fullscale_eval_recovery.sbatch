#!/usr/bin/env bash
# Sampling finished before the cluster-wide interruption; only evaluation is rerun.
#SBATCH --job-name=llmtd-news-e4k-eval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SOURCE="$A/outputs/news_e4000_fullscale_1094484"
OUT="$A/outputs/news_e4000_fullscale_eval_recovered_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff_e4000=$SOURCE/samples/unconditional.csv" \
  --sample "final_repair_e4000=$SOURCE/samples/final_repair.csv" \
  --sample "dynamic_e4000=$SOURCE/samples/dense_factor.csv" \
  --sample "dynamic_certified_e4000=$SOURCE/samples/dense_factor_then_certifier.csv" \
  --seed 20261024 --skip-detection --output "$OUT"

