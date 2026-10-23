#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dg-repaint-a
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID required}"; : "${EVAL_JOB_ID:?EVAL_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/aggregate_dynamic_graph_repaint.py" --outputs "$A/outputs" \
  --sample-job "$SAMPLE_JOB_ID" --eval-job "$EVAL_JOB_ID" \
  --output "$A/outputs/dg_repaint_aggregate_${SLURM_JOB_ID}"
