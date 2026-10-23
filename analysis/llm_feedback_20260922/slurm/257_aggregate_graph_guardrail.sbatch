#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dia-guard-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

A=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
: "${GUARD_JOB:?missing GUARD_JOB}"
"$PY" "$A/scripts/aggregate_graph_guardrail.py" --analysis "$A" --job "$GUARD_JOB" \
  --output "$A/outputs/graph_guardrail_aggregate_${SLURM_JOB_ID}"
