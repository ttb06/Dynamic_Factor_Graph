#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xdata-catalog
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${GRAPH_JOB_ID:?GRAPH_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(adult default)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
graph="$A/outputs/cross_dataset_complex_graph_${dataset}_${GRAPH_JOB_ID}_${SLURM_ARRAY_TASK_ID}/unified_graph.json"
output="$A/outputs/cross_dataset_catalog_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/curate_expanded_constraint_catalog.py" \
  --dataset "$dataset" --graph "$graph" --output "$output" \
  --min-support 60 --max-score-edges 12 --max-monitor-edges 8
