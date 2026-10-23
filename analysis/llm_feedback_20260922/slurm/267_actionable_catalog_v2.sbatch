#!/usr/bin/env bash
#SBATCH --job-name=llmtd-catalog-v2
#SBATCH --partition=cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${GRAPH_JOB_ID:?GRAPH_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers default diabetes); MINS=(60 60 100)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}; minimum=${MINS[$SLURM_ARRAY_TASK_ID]}
"$PY" "$A/scripts/curate_expanded_constraint_catalog.py" --dataset "$dataset" \
  --graph "$A/outputs/actionable_graph_v2_${dataset}_${GRAPH_JOB_ID}_${SLURM_ARRAY_TASK_ID}/unified_graph.json" \
  --min-support "$minimum" --max-score-edges 12 --max-monitor-edges 8 \
  --output "$A/outputs/actionable_catalog_v2_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
