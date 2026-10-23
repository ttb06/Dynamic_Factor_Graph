#!/usr/bin/env bash
# Re-evaluate an existing paired Adult TabDiff sample with the same main-table metrics.
#SBATCH --job-name=llmtd-adult-tabdiff-e
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset adult \
  --graph "$A/outputs/extended_bscg_adult_1081004/graph.json" \
  --sample "TabDiff=$A/outputs/extended_progressive_controller_adult_1081007/samples/unconditional.csv" \
  --seed 20261447 --max-rows 4096 \
  --output "$A/outputs/adult_tabdiff_matched_eval_${SLURM_JOB_ID}"

