#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-meaningful
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
MODELS=(gpt-5.6-luna gpt-5.6-terra)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SAFE_MODEL=${MODEL//./_}
OUT="$A/outputs/news_meaningful_typed_${SAFE_MODEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "$REPO"
"$PY" "$A/scripts/meaningful_typed_constraint_discovery.py" \
  --repo "$REPO" --dataset news --output "$OUT" \
  --env-file "$ROOT/.env" --llm-review --model "$MODEL" \
  --max-output-tokens 10000 --max-candidates 24 --seed 20261017 \
  --documented-graph "$A/outputs/news_documented_semantic_graph_20261016/graph.json" \
  --old-empirical-graph "$A/outputs/news_semantic_program_1085296/empirical_graph.json"
