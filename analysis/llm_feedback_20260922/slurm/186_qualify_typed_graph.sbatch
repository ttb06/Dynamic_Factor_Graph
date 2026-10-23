#!/usr/bin/env bash
#SBATCH --job-name=llmtd-typed-qualify
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CURATE_JOB_ID:?CURATE_JOB_ID required}"; : "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/qualify_typed_graph.py" \
  --curation "$A/outputs/typed_graph_curation_${CURATE_JOB_ID}" \
  --trajectory "$A/outputs/typed_graph_trajectory_${TRAJECTORY_JOB_ID}" \
  --output "$A/outputs/typed_graph_qualification_${SLURM_JOB_ID}"
