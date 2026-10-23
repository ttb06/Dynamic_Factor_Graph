#!/usr/bin/env bash
#SBATCH --job-name=llmtd-complex-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${RUN_JOB_ID:?RUN_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/audit_complex_graph_protocol.py" \
  --run "$A/outputs/complex_graph_protocol_${RUN_JOB_ID}" \
  --lcats-edges "$A/outputs/lcats_graph_audit_20261003/lcats_graph_edges.csv" \
  --output "$A/outputs/complex_graph_audit_${SLURM_JOB_ID}"
