#!/usr/bin/env bash
#SBATCH --job-name=llmtd-lcats-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
cd "$REPO"
"$ROOT/baseline_runs/envs/grdm/bin/python" "$A/scripts/audit_lcats_official_outputs.py" \
  --lcats-root "$ROOT/external/LCATS" --dataset insurance --target charges \
  --output "$A/outputs/lcats_official_output_audit_${SLURM_JOB_ID}"
