#!/bin/bash
#SBATCH --job-name=llmtd-graph-change
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
R=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff; A="$R/analysis/llm_feedback_20260922"; cd "$R"
/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python "$A/scripts/temporal_graph_change_detection.py" --repo "$R" --lifecycle "$A/outputs/temporal_graph_lifecycle_1051556" --output "$A/outputs/temporal_graph_change_${SLURM_JOB_ID}"
