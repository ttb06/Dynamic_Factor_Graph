#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dset-inventory
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SCRIPT="$A/scripts/inventory_multidataset_assets.py"
OUT="$A/outputs/multidataset_inventory_${SLURM_JOB_ID}"

"$PY" -u "$SCRIPT" --repo "$REPO" --analysis "$A" --output "$OUT"
{
  printf 'stage=multidataset_asset_inventory\n'
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'host=%s\n' "$(hostname)"
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  sha256sum "$SCRIPT"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/multidataset_inventory_${SLURM_JOB_ID}.pointer"
