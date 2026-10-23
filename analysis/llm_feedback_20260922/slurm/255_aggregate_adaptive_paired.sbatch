#!/usr/bin/env bash
#SBATCH --job-name=llmtd-adaptive-pair-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

A=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
: "${PAIRED_JOB:?missing PAIRED_JOB}"
"$PY" "$A/scripts/aggregate_adaptive_paired.py" --analysis "$A" --job "$PAIRED_JOB" \
  --output "$A/outputs/adaptive_paired_aggregate_${SLURM_JOB_ID}"
