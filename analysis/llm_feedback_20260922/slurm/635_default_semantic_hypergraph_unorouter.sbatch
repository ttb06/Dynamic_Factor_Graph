#!/usr/bin/env bash
#SBATCH --job-name=llmtd-semgraph
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
ENV_FILE="$ROOT/.env"
EXISTING="$A/outputs/bootstrap_conditional_graph_default_1070443_0/graph.json"

cd "$REPO"
"$PY" "$A/scripts/knowledge_graph_protocol.py" \
  --repo "$REPO" --dataset default --env-file "$ENV_FILE" \
  --proposal-model gpt-5.6-luna --external-model gpt-5.6-luna \
  --critic-model gpt-5.6-luna --judge-model gpt-5.6-luna \
  --proposal-profile connectivity_aware --existing-graph "$EXISTING" \
  --reasoning-effort none --judge-reasoning-effort none \
  --rounds 2 --per-role 10 --min-support 60 --seed 20261501 \
  --critic-batch-size 8 --judge-batch-size 8 \
  --max-review-candidates 48 --max-judge-candidates 36 \
  --max-output-tokens 6000 --total-token-budget 180000 \
  --output "$A/outputs/default_semantic_hypergraph_unorouter_${SLURM_JOB_ID}"
