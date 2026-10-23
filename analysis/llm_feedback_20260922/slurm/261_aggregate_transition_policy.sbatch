#!/usr/bin/env bash
#SBATCH --job-name=llmtd-transition-policy-agg
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${SOURCE_JOB_ID:?SOURCE_JOB_ID required}"

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/aggregate_transition_policy.py" \
  --analysis "$A" --job "$SOURCE_JOB_ID" \
  --output "$A/outputs/transition_policy_aggregate_${SLURM_JOB_ID}"
