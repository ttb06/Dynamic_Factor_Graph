#!/usr/bin/env bash
# Resume the cached few-shot graph protocol after a transient empty router
# response.  Successful prior stages are read from api_calls.jsonl and are not
# billed or executed again.
#SBATCH --job-name=llmtd-news-semgraph-resume
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/knowledge_graph_protocol.py" \
  --repo "$REPO" --dataset news --env-file "$ROOT/.env" \
  --proposal-model gpt-5.6-luna --external-model gpt-5.6-luna \
  --critic-model gpt-5.6-luna --judge-model gpt-5.6-luna \
  --proposal-profile semantic_programmatic --rounds 2 --per-role 8 \
  --max-review-candidates 36 --max-judge-candidates 24 \
  --critic-batch-size 8 --judge-batch-size 12 \
  --max-output-tokens 4000 --total-token-budget 180000 \
  --min-support 40 --resume \
  --output "$A/outputs/news_semantic_program_1085296"
