#!/usr/bin/env bash
#SBATCH --job-name=llmtd-shop-kg-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${RUN_JOB_ID:?RUN_JOB_ID must identify the Shoppers graph job}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
run="$A/outputs/shoppers_budgeted_graph_${RUN_JOB_ID}"
output="$A/outputs/shoppers_budgeted_graph_audit_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/analyze_shoppers_graph_hypotheses.py" --run "$run" --output "$output"
