#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cuts-de-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
cd "$REPO"
"$ROOT/baseline_runs/envs/grdm/bin/python" "$A/scripts/audit_cuts_official_generic.py" \
  --cuts-root "$ROOT/external/cuts" --dataset German --output "$A/outputs/cuts_official_german_audit_${SLURM_JOB_ID}"
