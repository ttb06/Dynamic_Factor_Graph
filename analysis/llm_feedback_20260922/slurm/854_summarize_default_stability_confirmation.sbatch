#!/usr/bin/env bash
# CPU-only fail-closed summary for gt4 fresh-pool confirmation.
#SBATCH --job-name=llmtd-gt4-summary
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

: "${CONFIRM_JOB:?missing CONFIRM_JOB}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
A="$ROOT/baseline_repos/tabdiff/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SUMMARIZER="$A/scripts/summarize_default_stability_confirmation.py"
OUT="$A/outputs/default_gt4_confirmation_${CONFIRM_JOB}_${SLURM_JOB_ID}"
"$PY" -u "$SUMMARIZER" --input-glob "$A/outputs/default_gt4_confirm_s*_${CONFIRM_JOB}_*" --output "$OUT"
{
  printf 'stage=Default_stability_gt4_fresh_confirmation_summary\n'
  printf 'confirm_job=%s\njob_id=%s\nhost=%s\nfinished_at=%s\n' "$CONFIRM_JOB" "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$SUMMARIZER"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/default_gt4_confirmation_${CONFIRM_JOB}.pointer"
