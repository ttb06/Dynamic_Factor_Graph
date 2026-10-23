#!/usr/bin/env bash
#SBATCH --job-name=llmtd-default-hq-cat
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${GRAPH_JOB_ID:?GRAPH_JOB_ID required}"
A=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/curate_expanded_constraint_catalog.py" --dataset default \
  --graph "$A/outputs/default_hq_graph_${GRAPH_JOB_ID}/unified_graph.json" --min-support 60 \
  --max-score-edges 12 --max-monitor-edges 8 --output "$A/outputs/default_hq_catalog_${SLURM_JOB_ID}"
