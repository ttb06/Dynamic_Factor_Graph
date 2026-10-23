#!/usr/bin/env bash
#SBATCH --job-name=llmtd-default-hq-union
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CATALOG_JOB_ID:?CATALOG_JOB_ID required}"
A=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/build_union_evaluation_graph.py" \
  --old "$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json" \
  --new "$A/outputs/default_hq_catalog_${CATALOG_JOB_ID}/constraint_catalog.json" \
  --output "$A/outputs/default_hq_union_${SLURM_JOB_ID}"
