#!/usr/bin/env bash
#SBATCH --job-name=llmtd-connect-edge-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${VARIANT_JOB_ID:?VARIANT_JOB_ID required}"
: "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
: "${GUIDANCE_JOB_ID:?GUIDANCE_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/aggregate_connectivity_experiments.py" --analysis "$A" \
  --variant-job "$VARIANT_JOB_ID" --trajectory-job "$TRAJECTORY_JOB_ID" \
  --guidance-job "$GUIDANCE_JOB_ID" --prefix connectivity_edge_schedule \
  --output "$A/outputs/connectivity_edge_aggregate_${SLURM_JOB_ID}"
