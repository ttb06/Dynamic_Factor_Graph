#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-typed-loop
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$A/outputs/news_llm_typed_completion_${SLURM_JOB_ID}"

cd "$REPO"
"$PY" "$A/scripts/meaningful_typed_constraint_discovery.py" \
  --repo "$REPO" --dataset news --output "$OUT" \
  --env-file "$ROOT/.env" --llm-propose --llm-review --model gpt-5.6-luna \
  --max-output-tokens 10000 --max-candidates 24 --seed 20261017 \
  --documented-graph "$A/outputs/news_documented_semantic_graph_20261016/graph.json" \
  --old-empirical-graph "$A/outputs/news_semantic_program_1085296/empirical_graph.json"
