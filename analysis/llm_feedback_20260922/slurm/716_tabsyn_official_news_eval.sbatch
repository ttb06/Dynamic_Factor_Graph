#!/usr/bin/env bash
# Common evaluation for the official TabSyn News run.  This is separate from
# training so a completed sample can always be re-evaluated without retraining.
#SBATCH --job-name=llmtd-tabsyn24-news-e
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${TABSYN_NEWS_JOB_ID:?TABSYN_NEWS_JOB_ID required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SAMPLE="$A/outputs/tabsyn_official_news_${TABSYN_NEWS_JOB_ID}/tabsyn_4096.csv"
GRAPH="$A/outputs/news_constraint_injection_variants_v2_20261017/order_plus_consensus_soft.json"
OUT="$A/outputs/tabsyn_official_news_eval_${SLURM_JOB_ID}"

cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset news --graph "$GRAPH" \
  --sample "tabsyn_official=$SAMPLE" --seed 20261017 --max-rows 4096 \
  --output "$OUT"
