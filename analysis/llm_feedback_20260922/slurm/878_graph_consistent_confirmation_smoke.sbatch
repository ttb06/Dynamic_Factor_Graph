#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-conf-smoke
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

bash -n "$A/slurm/876_confirm_graph_consistent_winners.sbatch"
bash -n "$A/slurm/877_summarize_graph_consistent_confirmation.sbatch"
"$PY" -m py_compile "$A/scripts/summarize_graph_consistent_confirmation.py"
"$PY" -u "$A/scripts/summarize_graph_consistent_confirmation.py" --help >/dev/null
sha256sum \
  "$A/scripts/summarize_graph_consistent_confirmation.py" \
  "$A/slurm/876_confirm_graph_consistent_winners.sbatch" \
  "$A/slurm/877_summarize_graph_consistent_confirmation.sbatch"
