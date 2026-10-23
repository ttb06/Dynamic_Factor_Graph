#!/usr/bin/env bash
#SBATCH --job-name=llmtd-probagg
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; ANALYSIS="$REPO/analysis/llm_feedback_20260922"; RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"; OUT="$RUN_ROOT/prob_gate_aggregate_${SLURM_JOB_ID}"
cd "$REPO"; "$PY" "$ANALYSIS/scripts/probabilistic_gate_control_aggregate.py" --input-root "$RUN_ROOT" --output "$OUT"
