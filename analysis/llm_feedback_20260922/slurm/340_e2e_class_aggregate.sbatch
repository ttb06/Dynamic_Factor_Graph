#!/usr/bin/env bash
#SBATCH --job-name=llmtd-e2e-class-agg
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${TRAIN_JOB_ID:?TRAIN_JOB_ID required}"; : "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"; : "${EVALUATION_JOB_ID:?EVALUATION_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"; PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/aggregate_e2e_class_objective.py" --analysis "$A" --train-job "$TRAIN_JOB_ID" \
  --trajectory-job "$TRAJECTORY_JOB_ID" --evaluation-job "$EVALUATION_JOB_ID" \
  --output "$A/outputs/e2e_class_objective_${SLURM_JOB_ID}"
