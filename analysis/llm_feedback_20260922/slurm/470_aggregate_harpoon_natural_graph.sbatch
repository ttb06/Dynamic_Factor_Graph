#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-aggregate
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python

"$PY" "$A/scripts/aggregate_harpoon_natural_graph.py" \
  --runs \
    "$A/outputs/harpoon_natural_graph_full_v3_s20261376_20261010" \
    "$A/outputs/harpoon_natural_graph_full_v3_s20261377_20261010" \
    "$A/outputs/harpoon_natural_graph_full_v3_s20261378_20261010" \
  --dynamic-summary "$A/outputs/natural_dag_stable_confirmation_v2_20261010/summary_metrics.csv" \
  --output "$A/outputs/harpoon_natural_graph_aggregate_v3_20261010"
