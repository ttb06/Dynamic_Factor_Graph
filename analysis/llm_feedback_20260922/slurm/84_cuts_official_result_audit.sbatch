#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cutsaudit
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/cuts/bin/python"
cd "$REPO"
"$PY" "$A/scripts/cuts_official_result_audit.py" --cuts-root "$ROOT/external/cuts" --output "$A/outputs/cuts_official_result_audit_${SLURM_JOB_ID}"
