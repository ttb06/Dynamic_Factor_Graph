#!/bin/bash
#SBATCH --job-name=scoped-agg
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
ANALYSIS="$ROOT/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
OUT="$RUN_ROOT/scoped_actuation_aggregate_${SLURM_JOB_ID}"
cd "$ROOT"
"$PY" "$ANALYSIS/scripts/scoped_actuation_aggregate.py" --input-root "$RUN_ROOT" --output "$OUT"
