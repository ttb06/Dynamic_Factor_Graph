#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-guard-build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${DISCOVERY_JOB_ID:?DISCOVERY_JOB_ID required}"
: "${AUDIT_JOB_ID:?AUDIT_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/build_llm_score_guardrail_graph.py" \
  --public-graph "$A/outputs/news_public_typed_graph_20261015/graph.json" \
  --llm-run "$A/outputs/news_complex_discovery_${DISCOVERY_JOB_ID}" \
  --audit "$A/outputs/news_complex_discovery_audit_${AUDIT_JOB_ID}" \
  --public-schedule "$A/outputs/news_public_dynamic_trajectory_1081001/typed_readability_schedule.csv" \
  --soft-cutoff .50 --output "$A/outputs/news_llm_guardrail_graph_${SLURM_JOB_ID}"
