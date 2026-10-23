#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-pa-csmoke
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
A="$ROOT/baseline_repos/tabdiff/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
bash -n "$A/slurm/883_confirm_graph_pareto_winners.sbatch"
bash -n "$A/slurm/884_summarize_graph_pareto_confirmation.sbatch"
"$PY" -m py_compile \
  "$A/scripts/graph_pareto_anchored_resampling.py" \
  "$A/scripts/summarize_graph_pareto_confirmation.py"
sha256sum \
  "$A/slurm/883_confirm_graph_pareto_winners.sbatch" \
  "$A/slurm/884_summarize_graph_pareto_confirmation.sbatch" \
  "$A/scripts/graph_pareto_anchored_resampling.py" \
  "$A/scripts/summarize_graph_pareto_confirmation.py"
