#!/usr/bin/env bash
#SBATCH --job-name=llmtd-adaptive-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

A=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
: "${ADAPTIVE_JOB:?missing ADAPTIVE_JOB}"
"$PY" "$A/scripts/aggregate_adaptive_volatility.py" --analysis "$A" --adaptive-job "$ADAPTIVE_JOB" \
  --output "$A/outputs/adaptive_volatility_aggregate_${SLURM_JOB_ID}"
