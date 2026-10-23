#!/usr/bin/env bash
#SBATCH --job-name=llmtd-semgraph-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${GRAPH_JOB_ID:?GRAPH_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
"$PY" "$A/scripts/audit_complex_graph_protocol.py" \
  --run "$A/outputs/default_semantic_hypergraph_unorouter_${GRAPH_JOB_ID}" \
  --output "$A/outputs/default_semantic_hypergraph_audit_${SLURM_JOB_ID}"
