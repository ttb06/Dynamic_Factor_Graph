#!/usr/bin/env bash
#SBATCH --job-name=llmtd-guide-select
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${TUNE_JOB_ID:?TUNE_JOB_ID required}"
A=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/aggregate_guidance_strength.py" --outputs "$A/outputs" --job-id "$TUNE_JOB_ID" \
  --output "$A/outputs/guidance_strength_selection_${TUNE_JOB_ID}"
