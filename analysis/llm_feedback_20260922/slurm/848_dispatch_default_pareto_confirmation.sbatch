#!/usr/bin/env bash
# One-shot bounded dispatcher: zero or one fresh confirmation array.
#SBATCH --job-name=llmtd-pareto-dispatch
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

: "${SCREEN_JOB:?missing SCREEN_JOB}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
A="$ROOT/baseline_repos/tabdiff/analysis/llm_feedback_20260922"
POINTER="$A/outputs/default_pareto_selection_${SCREEN_JOB}.pointer"
SELECTION_ROOT=$(cat "$POINTER")
SELECTED="$SELECTION_ROOT/selected_candidates.txt"
COUNT=$(sed '/^[[:space:]]*$/d' "$SELECTED" | wc -l)
RECEIPT="$A/outputs/default_pareto_dispatch_${SCREEN_JOB}_${SLURM_JOB_ID}.txt"

if [[ "$COUNT" -eq 0 ]]; then
  {
    printf 'status=no_promotion\n'
    printf 'screen_job=%s\ndispatch_job=%s\nselection_root=%s\n' "$SCREEN_JOB" "$SLURM_JOB_ID" "$SELECTION_ROOT"
  } > "$RECEIPT"
  printf '%s\n' "$RECEIPT" > "$A/outputs/default_pareto_dispatch_${SCREEN_JOB}.pointer"
  exit 0
fi
if [[ "$COUNT" -lt 1 || "$COUNT" -gt 2 ]]; then
  echo "invalid selected candidate count: $COUNT" >&2
  exit 64
fi

CONFIRM_JOB=$(sbatch --parsable --export=ALL,SCREEN_JOB="$SCREEN_JOB" \
  "$A/slurm/849_confirm_default_pareto_particles.sbatch")
SUMMARY_JOB=$(sbatch --parsable --dependency=afterok:"$CONFIRM_JOB" \
  --export=ALL,SCREEN_JOB="$SCREEN_JOB",CONFIRM_JOB="$CONFIRM_JOB" \
  "$A/slurm/850_summarize_default_pareto_confirmation.sbatch")
{
  printf 'status=submitted\n'
  printf 'screen_job=%s\ndispatch_job=%s\nselection_root=%s\n' "$SCREEN_JOB" "$SLURM_JOB_ID" "$SELECTION_ROOT"
  printf 'selected_count=%s\nconfirm_job=%s\nsummary_job=%s\n' "$COUNT" "$CONFIRM_JOB" "$SUMMARY_JOB"
} > "$RECEIPT"
printf '%s\n' "$RECEIPT" > "$A/outputs/default_pareto_dispatch_${SCREEN_JOB}.pointer"
