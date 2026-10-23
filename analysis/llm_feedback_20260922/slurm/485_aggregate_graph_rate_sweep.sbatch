#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-rate-agg
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python

"$PY" "$A/scripts/aggregate_graph_rate_sweep.py" \
  --baseline "$A/outputs/mixed_graph_splitmedian_full_continue_s20261376_1067476_3" \
  --runs \
    "$A/outputs/graph_rate_median_graph_1067504_0" \
    "$A/outputs/graph_rate_upper_graph_1067504_1" \
    "$A/outputs/graph_rate_splitwilson025_graph_1067504_2" \
    "$A/outputs/graph_rate_splitwilson050_graph_1067504_3" \
    "$A/outputs/graph_rate_median_late_1067504_4" \
    "$A/outputs/graph_rate_median_graph_late_1067504_5" \
  --output "$A/outputs/graph_rate_controller_selection_20261010"
