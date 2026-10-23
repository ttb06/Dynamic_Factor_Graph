#!/usr/bin/env bash
# CPU-only aggregation for the six-seed UAR confirmation wave.
# Submit with --export=ALL,CONFIRM_JOB=<array_job_id>.
#SBATCH --job-name=llmtd-news-uar-summary
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

: "${CONFIRM_JOB:?submit with --export=ALL,CONFIRM_JOB=<array_job_id>}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SUMMARIZER="$A/scripts/summarize_uar_confirmation.py"
OUT="$A/outputs/news_uar_confirmation_summary_${CONFIRM_JOB}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

"$PY" -u "$SUMMARIZER" \
  --input-glob "$A/outputs/news_uar_confirm_*_${CONFIRM_JOB}_*/eval/metrics.csv" \
  --output "$OUT"

{
  printf 'stage=confirmation_summary\n'
  printf 'confirm_job=%s\n' "$CONFIRM_JOB"
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'host=%s\n' "$(hostname)"
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  sha256sum "$SUMMARIZER"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" \
  > "$A/outputs/news_uar_confirmation_summary_${CONFIRM_JOB}.pointer"
