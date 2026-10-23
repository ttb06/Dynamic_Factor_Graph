#!/usr/bin/env bash
# CPU-only fail-closed selector for integrated train-coverage screen.
#SBATCH --job-name=llmtd-intcov-select
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
SELECTOR="$A/scripts/select_default_geocoverage_particles.py"
OUT="$A/outputs/default_intcov_selection_${SCREEN_JOB}_${SLURM_JOB_ID}"
"$PY" -u "$SELECTOR" --round integrated \
  --input-glob "$A/outputs/default_intcov_*_s*_${SCREEN_JOB}_*" --output "$OUT"
{
  printf 'stage=Default_integrated_coverage_particle_selector\n'
  printf 'screen_job=%s\njob_id=%s\nhost=%s\nfinished_at=%s\n' "$SCREEN_JOB" "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$SELECTOR"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/default_intcov_selection_${SCREEN_JOB}.pointer"
