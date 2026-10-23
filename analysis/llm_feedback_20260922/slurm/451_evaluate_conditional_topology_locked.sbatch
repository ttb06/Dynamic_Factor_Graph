#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cond-lock-eval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

: "${SAMPLING_JOB_ID:?SAMPLING_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/evaluate_conditional_topology_controls.py" \
  --root "$A/outputs" --job-id "$SAMPLING_JOB_ID" \
  --output "$A/outputs/conditional_topology_locked_paired_audit_${SAMPLING_JOB_ID}_${SLURM_JOB_ID}"
