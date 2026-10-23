#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-disc-resume
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${SOURCE_JOB_ID:?SOURCE_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/knowledge_graph_protocol.py" \
  --repo "$REPO" --dataset news --env-file "$ROOT/.env" \
  --proposal-model gpt-5.6-luna --external-model gpt-5.6-luna \
  --critic-model gpt-5.6-luna --judge-model gpt-5.6-luna \
  --proposal-profile complex_hyperedge --rounds 2 --per-role 8 \
  --max-review-candidates 32 --max-judge-candidates 24 \
  --critic-batch-size 8 --judge-batch-size 12 \
  --max-output-tokens 3500 --total-token-budget 240000 \
  --min-support 40 --resume \
  --output "$A/outputs/news_complex_discovery_${SOURCE_JOB_ID}"
