#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-pq-csmoke
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
A="$ROOT/baseline_repos/tabdiff/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
bash -n "$A/slurm/895_confirm_graph_proximal_quality_winners.sbatch"
bash -n "$A/slurm/896_summarize_graph_proximal_quality_confirmation.sbatch"
"$PY" -m py_compile "$A/scripts/graph_proximal_swap_repair.py" \
  "$A/scripts/summarize_graph_pareto_confirmation.py"
