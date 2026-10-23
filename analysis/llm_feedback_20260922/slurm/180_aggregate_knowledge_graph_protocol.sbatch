#!/usr/bin/env bash
#SBATCH --job-name=llmtd-kg-proto-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${RUN_JOB_ID:?RUN_JOB_ID must identify the protocol array job}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
mapfile -t inputs < <(find "$A/outputs" -maxdepth 1 -type d -name "knowledge_graph_protocol_*_${RUN_JOB_ID}_*" | sort)
test "${#inputs[@]}" -eq 3
output="$A/outputs/knowledge_graph_protocol_aggregate_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/aggregate_knowledge_graph_protocol.py" --inputs "${inputs[@]}" --output "$output"
