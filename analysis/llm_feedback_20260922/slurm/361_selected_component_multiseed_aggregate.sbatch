#!/usr/bin/env bash
#SBATCH --job-name=llmtd-selected-agg
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/miniforge3/bin/python
: "${PARENT_JOB_ID:?PARENT_JOB_ID is required}"
: "${STRENGTH_JOB_ID:?STRENGTH_JOB_ID is required}"

"$PY" "$A/scripts/aggregate_selected_component_multiseed.py" \
  --root "$A/outputs" --job-id "$PARENT_JOB_ID" \
  --original-component "$A/outputs/component_strength_s150_${STRENGTH_JOB_ID}_1" \
  --output "$A/outputs/selected_component_multiseed_${SLURM_JOB_ID}"
