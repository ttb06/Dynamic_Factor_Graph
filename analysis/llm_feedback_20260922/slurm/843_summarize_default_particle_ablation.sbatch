#!/usr/bin/env bash
# CPU-only aggregation of multi-seed particle-selection ablations.
#SBATCH --job-name=llmtd-particle-ablation-summary
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

: "${ABLATION_JOB:?submit with --export=ALL,ABLATION_JOB=<array_job_id>}"
if [[ ! "$ABLATION_JOB" =~ ^[0-9]+$ ]]; then
  echo "ABLATION_JOB must be numeric" >&2
  exit 64
fi

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SUMMARIZER="$A/scripts/summarize_default_particle_ablation.py"
OUT="$A/outputs/default_particle_ablation_${ABLATION_JOB}_${SLURM_JOB_ID}"

"$PY" -u "$SUMMARIZER" \
  --input-glob "$A/outputs/default_particle_ablation_s*_${ABLATION_JOB}_*" \
  --output "$OUT"

{
  printf 'stage=Default_balanced_particle_multiseed_ablation_summary\n'
  printf 'ablation_job=%s\njob_id=%s\nhost=%s\nfinished_at=%s\n' \
    "$ABLATION_JOB" "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$SUMMARIZER"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/default_particle_ablation_${ABLATION_JOB}.pointer"
