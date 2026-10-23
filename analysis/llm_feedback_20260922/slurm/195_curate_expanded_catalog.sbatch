#!/usr/bin/env bash
#SBATCH --job-name=llmtd-expand-catalog
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
cd "$REPO"
python "$A/scripts/curate_expanded_constraint_catalog.py" \
  --graph "$A/outputs/complex_graph_protocol_1056389/unified_graph.json" \
  --max-score-edges 8 --max-monitor-edges 5 \
  --output "$A/outputs/expanded_constraint_catalog_${SLURM_JOB_ID}"
