#!/usr/bin/env bash
# Same-sample comparison of the soft routed method with deterministic final
# repair and routed method + late certifier.  No generation is repeated.
#SBATCH --job-name=llmtd-news-route-repair
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SRC="$A/outputs/news_llm_route_strength_1092100_5_q035_s075/samples"
OUT="$A/outputs/news_soft_route_matched_repair_${SLURM_JOB_ID}"

cd "$REPO"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$SRC/unconditional.csv" \
  --sample "final_repair=$SRC/final_repair.csv" \
  --sample "soft_route=$SRC/dense_factor.csv" \
  --sample "soft_route_then_certifier=$SRC/dense_factor_then_certifier.csv" \
  --seed 20261607 --output "$OUT"
