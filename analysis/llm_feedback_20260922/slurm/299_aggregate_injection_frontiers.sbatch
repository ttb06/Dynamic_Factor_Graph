#!/usr/bin/env bash
#SBATCH --job-name=llmtd-inject-aggregate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
: "${FRONTIER_JOB_IDS:?FRONTIER_JOB_IDS is required}"
read -r -a ids <<< "$FRONTIER_JOB_IDS"
"$PY" "$A/scripts/aggregate_injection_frontiers.py" \
  --outputs "$A/outputs" --job-ids "${ids[@]}" \
  --output "$A/outputs/injection_frontiers_${SLURM_JOB_ID}"
