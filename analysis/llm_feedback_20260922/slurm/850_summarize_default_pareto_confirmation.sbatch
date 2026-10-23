#!/usr/bin/env bash
# CPU-only fail-closed summary for fresh-pool Pareto confirmation.
#SBATCH --job-name=llmtd-pareto-confirm-summary
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

: "${SCREEN_JOB:?missing SCREEN_JOB}"
: "${CONFIRM_JOB:?missing CONFIRM_JOB}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SUMMARIZER="$A/scripts/summarize_default_pareto_confirmation.py"
SELECTION_ROOT=$(cat "$A/outputs/default_pareto_selection_${SCREEN_JOB}.pointer")
OUT="$A/outputs/default_pareto_confirmation_${CONFIRM_JOB}_${SLURM_JOB_ID}"

"$PY" -u "$SUMMARIZER" \
  --input-glob "$A/outputs/default_pareto_confirm_s*_${CONFIRM_JOB}_*" \
  --selection-file "$SELECTION_ROOT/selected_candidates.txt" \
  --screen-job "$SCREEN_JOB" --output "$OUT"

{
  printf 'stage=Default_pareto_particle_fresh_confirmation_summary\n'
  printf 'screen_job=%s\nconfirm_job=%s\njob_id=%s\nhost=%s\nfinished_at=%s\n' \
    "$SCREEN_JOB" "$CONFIRM_JOB" "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$SUMMARIZER" "$SELECTION_ROOT/selection.json"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/default_pareto_confirmation_${CONFIRM_JOB}.pointer"
