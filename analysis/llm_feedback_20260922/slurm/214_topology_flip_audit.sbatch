#!/usr/bin/env bash
#SBATCH --job-name=llmtd-flip-audit
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${PAIR_JOB_ID:?PAIR_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers default)
CATALOGS=(expanded_constraint_catalog_1057680 cross_dataset_catalog_default_1058707_1)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}; catalog=${CATALOGS[$SLURM_ARRAY_TASK_ID]}
base="$A/outputs/topology_flip_pair_${dataset}_${PAIR_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/analyze_topology_flip_subgroup.py" \
  --dynamic "${base}_dynamic" --frozen "${base}_frozen" \
  --graph "$A/outputs/$catalog/constraint_catalog.json" \
  --output "$A/outputs/topology_flip_audit_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
