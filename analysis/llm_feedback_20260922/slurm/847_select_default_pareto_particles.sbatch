#!/usr/bin/env bash
# CPU-only fail-closed selector for the multi-seed Pareto-particle screen.
#SBATCH --job-name=llmtd-pareto-select
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
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
SELECTOR="$A/scripts/select_default_pareto_particles.py"
OUT="$A/outputs/default_pareto_selection_${SCREEN_JOB}_${SLURM_JOB_ID}"

"$PY" -u "$SELECTOR" \
  --input-glob "$A/outputs/default_pareto_*_s*_${SCREEN_JOB}_*" \
  --output "$OUT"

{
  printf 'stage=Default_pareto_particle_multiseed_selector\n'
  printf 'screen_job=%s\njob_id=%s\nhost=%s\nfinished_at=%s\n' \
    "$SCREEN_JOB" "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$SELECTOR"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/default_pareto_selection_${SCREEN_JOB}.pointer"
