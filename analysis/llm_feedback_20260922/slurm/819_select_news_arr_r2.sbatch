#!/usr/bin/env bash
# CPU-only aggregation and fail-closed promotion for News ARR R2.
# Submit with --export=ALL,R2_JOB=<array_job_id> afterok dependency.
#SBATCH --job-name=llmtd-news-arr-select
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

: "${R2_JOB:?submit with --export=ALL,R2_JOB=<array_job_id>}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SELECTOR="$A/scripts/select_psdfr_candidates.py"
OUT="$A/outputs/news_arr_r2_selection_${R2_JOB}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

"$PY" -u "$SELECTOR" \
  --input-glob "$A/outputs/news_arr_r2_*_${R2_JOB}_*/eval/metrics.csv" \
  --output "$OUT" --max-winners 2

{
  printf 'stage=R2_selection\n'
  printf 'r2_job=%s\n' "$R2_JOB"
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'host=%s\n' "$(hostname)"
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  sha256sum "$SELECTOR"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/news_arr_r2_selection_${R2_JOB}.pointer"
