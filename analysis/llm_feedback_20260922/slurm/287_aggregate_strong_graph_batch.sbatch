#!/usr/bin/env bash
#SBATCH --job-name=llmtd-strong-aggregate
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CONTROL_JOB:?CONTROL_JOB is required}"
: "${COMBINED_JOB:?COMBINED_JOB is required}"
A=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/aggregate_strong_graph_batch.py" \
  --outputs "$A/outputs" --control-job "$CONTROL_JOB" --combined-job "$COMBINED_JOB" \
  --output "$A/outputs/strong_graph_batch_${SLURM_JOB_ID}"
