#!/usr/bin/env bash
#SBATCH --job-name=llmtd-repaint-boot
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
mid="$A/outputs/dg_repaint_shoppers_mid25_broad_1070752_3/samples/graph_repaint_mid_plus_repair.csv"
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
  base="$A/outputs/dg_repaint_shoppers_mid25_broad_1070752_3/samples/calibrated_final_repair.csv"
  name=calibrated_final_repair
else
  base="$A/outputs/dg_repaint_final_broad_shoppers_1070793_0/samples/graph_repaint_final_plus_repair.csv"
  name=matched_final_broad_repaint
fi
cd "$REPO"
"$PY" "$A/scripts/paired_graph_error_bootstrap.py" \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --heldout "$REPO/synthetic/shoppers/test.csv" --candidate "$mid" --baseline "$base" \
  --candidate-name mid_broad_repaint_plus_repair --baseline-name "$name" --replicates 5000 \
  --output "$A/outputs/repaint_bootstrap_${name}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
