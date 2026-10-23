#!/bin/bash
#SBATCH --job-name=llmtd-graph-life
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
cd "$REPO"
OUT="$REPO/analysis/llm_feedback_20260922/outputs/temporal_graph_lifecycle_${SLURM_JOB_ID}"
/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python analysis/llm_feedback_20260922/scripts/temporal_graph_lifecycle.py \
  --repo "$REPO" --output "$OUT" --datasets beijing news shoppers
