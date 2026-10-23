#!/usr/bin/env bash
#SBATCH --job-name=llmtd-hyper-minimal
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
"$PY" "$A/scripts/hyperedge_minimality_audit.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/connectivity_graph_generation_1062182/llm_protocol_graph.json" \
  --seed 20261222 --min-new-support 30 --min-rate-increase .01 \
  --output "$A/outputs/hyperedge_minimality_${SLURM_JOB_ID}"
