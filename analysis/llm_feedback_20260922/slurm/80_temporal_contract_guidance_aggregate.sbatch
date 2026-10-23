#!/usr/bin/env bash
#SBATCH --job-name=llmtd-tempagg
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/temporal_contract_guidance_aggregate.py" --input-root "$R" --output "$R/temporal_contract_guidance_aggregate_${SLURM_JOB_ID}"
