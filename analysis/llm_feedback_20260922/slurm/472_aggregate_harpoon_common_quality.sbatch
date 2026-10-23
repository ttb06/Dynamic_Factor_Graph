#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-quality-agg
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

"$PY" "$A/scripts/aggregate_matched_common_quality.py" \
  --inputs \
    "$A/outputs/harpoon_natural_quality_s20261376_1067018_0/baseline_metrics.csv" \
    "$A/outputs/harpoon_natural_quality_s20261377_1067018_1/baseline_metrics.csv" \
    "$A/outputs/harpoon_natural_quality_s20261378_1067018_2/baseline_metrics.csv" \
  --output "$A/outputs/harpoon_natural_common_quality_aggregate_v3_20261010"
