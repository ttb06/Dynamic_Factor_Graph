#!/usr/bin/env bash
#SBATCH --job-name=llmtd-solver-router
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"; "$PY" "$A/scripts/dynamic_solver_router_analysis.py" --default "$A/outputs/dynamic_graph_solver_default_1051370_0" --shoppers "$A/outputs/dynamic_graph_solver_shoppers_1051370_1" --output "$A/outputs/dynamic_solver_router_${SLURM_JOB_ID}"
