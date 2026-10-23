#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:08:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"

bash -n "$A/slurm/873_graph_consistent_screen.sbatch"
bash -n "$A/slurm/874_summarize_graph_consistent_screen.sbatch"
"$PY" -m py_compile \
  "$A/scripts/graph_consistent_particle_resampling.py" \
  "$A/scripts/audit_graph_consistency.py" \
  "$A/scripts/summarize_graph_consistent_screen.py"
"$PY" -u "$A/scripts/graph_consistent_particle_resampling.py" --help >/dev/null
"$PY" -u "$A/scripts/audit_graph_consistency.py" --help >/dev/null
"$PY" -u "$A/scripts/summarize_graph_consistent_screen.py" --help >/dev/null
sha256sum \
  "$A/scripts/graph_consistent_particle_resampling.py" \
  "$A/scripts/audit_graph_consistency.py" \
  "$A/scripts/summarize_graph_consistent_screen.py" \
  "$A/slurm/873_graph_consistent_screen.sbatch" \
  "$A/slurm/874_summarize_graph_consistent_screen.sbatch"
