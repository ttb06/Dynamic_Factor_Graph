#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-connect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/build_connectivity_graph_variants.py" \
  --source "$A/outputs/actionable_graph_v2_shoppers_1060952_0/unified_graph.json" \
  --base "$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json" \
  --complex-source "$A/outputs/complex_graph_protocol_1056389/llm_protocol_graph.json" \
  --min-support 60 --output "$A/outputs/connectivity_variants_${SLURM_JOB_ID}"
