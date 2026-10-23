#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-offagg
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
: "${SOURCE_JOB_ID:?SOURCE_JOB_ID is required}"
"$PY" "$A/scripts/aggregate_harpoon_official.py" \
  --root "$A/outputs" --source-job "$SOURCE_JOB_ID" \
  --output "$A/outputs/harpoon_official_5trial_aggregate_${SLURM_JOB_ID}"
