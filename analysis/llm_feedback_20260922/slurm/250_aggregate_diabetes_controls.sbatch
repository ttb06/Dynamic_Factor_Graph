#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dia-controls-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

A=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
: "${HYSTERESIS_JOB:?missing HYSTERESIS_JOB}"
: "${TOPOLOGY_JOB:?missing TOPOLOGY_JOB}"
: "${TIMING_JOB:?missing TIMING_JOB}"
"$PY" "$A/scripts/aggregate_diabetes_controls.py" --analysis "$A" \
  --hysteresis-job "$HYSTERESIS_JOB" --topology-job "$TOPOLOGY_JOB" --timing-job "$TIMING_JOB" \
  --output "$A/outputs/diabetes_controls_aggregate_${SLURM_JOB_ID}"
