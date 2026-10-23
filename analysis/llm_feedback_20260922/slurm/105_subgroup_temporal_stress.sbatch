#!/usr/bin/env bash
#SBATCH --job-name=llmtd-subgraph-time
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$A/outputs/subgroup_temporal_stress_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/subgroup_temporal_stress.py" \
  --repo "$REPO" --audit "$A/outputs/subgroup_graph_audit_1050323" \
  --output "$OUT" --min-cell 30 --min-interaction 0.025
