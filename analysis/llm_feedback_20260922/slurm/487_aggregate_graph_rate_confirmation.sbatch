#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-rate-confirm-agg
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

"$PY" "$A/scripts/aggregate_graph_rate_confirmation.py" \
  --baseline-runs \
    "$A/outputs/mixed_graph_splitmedian_full_continue_s20261376_1067476_3" \
    "$A/outputs/mixed_graph_splitmedian_full_continue_s20261377_1067476_4" \
    "$A/outputs/mixed_graph_splitmedian_full_continue_s20261378_1067476_5" \
  --graph-rate-runs \
    "$A/outputs/graph_rate_median_graph_1067504_0" \
    "$A/outputs/graph_rate_confirm_s20261377_1067511_0" \
    "$A/outputs/graph_rate_confirm_s20261378_1067511_1" \
  --output "$A/outputs/graph_rate_confirmation_20261010"
