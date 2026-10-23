#!/usr/bin/env bash
# Common evaluation for ForestDiffusion News.
#SBATCH --job-name=llmtd-forest24-news-e
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${FOREST_NEWS_JOB_ID:?FOREST_NEWS_JOB_ID required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SAMPLE="$A/outputs/forestdiffusion_news_official_k100_${FOREST_NEWS_JOB_ID}/forest_diffusion.csv"
GRAPH="$A/outputs/news_constraint_injection_variants_v2_20261017/order_plus_consensus_soft.json"
OUT="$A/outputs/forestdiffusion_news_official_eval_${SLURM_JOB_ID}"

cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset news --graph "$GRAPH" \
  --sample "forestdiffusion_official=$SAMPLE" --seed 20261017 --max-rows 4096 \
  --output "$OUT"
