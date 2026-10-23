#!/usr/bin/env bash
#SBATCH --job-name=llmtd-semantic-resume
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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
  --repo "$REPO" --dataset default --env-file "$ROOT/.env" \
  --proposal-model gpt-5.6-luna --external-model gpt-5.6-luna \
  --critic-model gpt-5.6-luna --judge-model gpt-5.6-luna \
  --proposal-profile trajectory_actionable --reasoning-effort none --judge-reasoning-effort none \
  --rounds 1 --per-role 4 --min-support 60 --seed 20261523 \
  --critic-batch-size 4 --judge-batch-size 4 \
  --max-review-candidates 24 --max-judge-candidates 18 \
  --max-output-tokens 3000 --total-token-budget 120000 --resume \
  --output "$A/outputs/default_semantic_focused_${SOURCE_JOB_ID}"
