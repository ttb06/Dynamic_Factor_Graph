#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cond-quality
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

: "${SAMPLING_JOB_ID:?SAMPLING_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
extra=()
if [[ -n "${LATE_JOB_ID:-}" ]]; then extra+=(--late-job-id "$LATE_JOB_ID"); fi
"$PY" "$A/scripts/evaluate_conditional_distribution_quality.py" --repo "$REPO" \
  --root "$A/outputs" --job-id "$SAMPLING_JOB_ID" \
  "${extra[@]}" \
  --output "$A/outputs/conditional_distribution_quality_${SAMPLING_JOB_ID}_${SLURM_JOB_ID}"
