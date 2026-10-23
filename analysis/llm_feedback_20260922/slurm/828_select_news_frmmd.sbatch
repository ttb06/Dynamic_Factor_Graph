#!/usr/bin/env bash
# CPU-only selector for News factorized RFF-MMD screen.
#SBATCH --job-name=llmtd-frmmd-select
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

: "${SCREEN_JOB:?submit with --export=ALL,SCREEN_JOB=<array_job_id>}"
if [[ ! "$SCREEN_JOB" =~ ^[0-9]+$ ]]; then
  echo "SCREEN_JOB must be numeric" >&2
  exit 64
fi

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SELECTOR="$A/scripts/select_frmmd_candidates.py"
OUT="$A/outputs/news_frmmd_selection_${SCREEN_JOB}_${SLURM_JOB_ID}"

"$PY" -u "$SELECTOR" \
  --input-glob "$A/outputs/news_frmmd_*_${SCREEN_JOB}_*" \
  --output "$OUT" --max-winners 2

{
  printf 'stage=News_factorized_RFF_MMD_selection\n'
  printf 'screen_job=%s\njob_id=%s\nhost=%s\nfinished_at=%s\n' \
    "$SCREEN_JOB" "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$SELECTOR"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/news_frmmd_selection_${SCREEN_JOB}.pointer"
