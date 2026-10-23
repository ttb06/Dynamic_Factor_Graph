#!/usr/bin/env bash
#SBATCH --job-name=llmtd-generator-blindspot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/audit_generator_blindspot_edges.py" \
  --graph "$A/outputs/cross_dataset_complex_graph_default_1057835_1/unified_graph.json" \
  --samples "$A/outputs/default_middle_state_projection_min25_1063029_0/samples/unconditional.csv" \
  --decisions "$A/outputs/cross_dataset_catalog_default_1058707_1/curation_decisions.json" \
  --output "$A/outputs/default_generator_blindspot_audit_${SLURM_JOB_ID}"
