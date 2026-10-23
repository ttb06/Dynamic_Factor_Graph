#!/usr/bin/env bash
# CPU-only fail-closed summary for News zero-shot particle transfer.
#SBATCH --job-name=llmtd-news-particle-summary
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

: "${SCREEN_JOB:?missing SCREEN_JOB}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
A="$ROOT/baseline_repos/tabdiff/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SUMMARIZER="$A/scripts/summarize_news_particle_transfer.py"
OUT="$A/outputs/news_particle_transfer_summary_${SCREEN_JOB}_${SLURM_JOB_ID}"
"$PY" -u "$SUMMARIZER" --input-glob "$A/outputs/news_particle_transfer_s*_${SCREEN_JOB}_*" --output "$OUT"
{
  printf 'stage=News_zero_shot_particle_transfer_summary\n'
  printf 'screen_job=%s\njob_id=%s\nhost=%s\nfinished_at=%s\n' "$SCREEN_JOB" "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$SUMMARIZER"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/news_particle_transfer_summary_${SCREEN_JOB}.pointer"
