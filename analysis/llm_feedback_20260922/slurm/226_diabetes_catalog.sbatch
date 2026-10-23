#!/usr/bin/env bash
#SBATCH --job-name=llmtd-diabetes-catalog
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${GRAPH_JOB_ID:?GRAPH_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/curate_expanded_constraint_catalog.py" --dataset diabetes \
  --graph "$A/outputs/diabetes_complex_graph_${GRAPH_JOB_ID}/unified_graph.json" \
  --min-support 100 --max-score-edges 16 --max-monitor-edges 10 \
  --output "$A/outputs/diabetes_catalog_${SLURM_JOB_ID}"
