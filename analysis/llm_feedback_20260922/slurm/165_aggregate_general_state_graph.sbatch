#!/usr/bin/env bash
#SBATCH --job-name=llmtd-state-graph-agg
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
: "${STATE_GRAPH_ARRAY_JOB:?STATE_GRAPH_ARRAY_JOB must be set}"
cd "$REPO"
"$PY" "$A/scripts/aggregate_general_state_graph.py" \
  --root "$A/outputs" --array-job "$STATE_GRAPH_ARRAY_JOB" \
  --output "$A/outputs/general_state_graph_aggregate_${SLURM_JOB_ID}"
