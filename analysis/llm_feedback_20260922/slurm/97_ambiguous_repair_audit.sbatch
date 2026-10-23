#!/usr/bin/env bash
#SBATCH --job-name=llmtd-ambrepair
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/tabdiff/bin/python"
OUT="${ANALYSIS_RUN_ROOT:-$A/outputs}/ambiguous_repair_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/ambiguous_repair_audit.py" --repo "$REPO" --output "$OUT" --seed 20260929
