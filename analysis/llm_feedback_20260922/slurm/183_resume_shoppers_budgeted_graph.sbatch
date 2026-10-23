#!/usr/bin/env bash
#SBATCH --job-name=llmtd-shop-kg-resume
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
ENV_FILE=/home/cs.aau.dk/tungkvt/binhtt2/EIPC/.env
output="$A/outputs/shoppers_budgeted_graph_1056016"
cd "$REPO"
"$PY" "$A/scripts/knowledge_graph_protocol.py" \
  --repo "$REPO" --dataset shoppers --env-file "$ENV_FILE" \
  --proposal-model gpt-4.1-mini --external-model gpt-5.4 \
  --critic-model gpt-4.1-mini --judge-model gpt-5.4 \
  --reasoning-effort none --judge-reasoning-effort none \
  --rounds 2 --per-role 7 --min-support 30 --seed 20261005 \
  --judge-batch-size 10 --max-output-tokens 4500 --total-token-budget 165000 \
  --resume --output "$output"
