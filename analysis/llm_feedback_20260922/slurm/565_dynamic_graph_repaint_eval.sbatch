#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dg-repaint-e
#SBATCH --partition=cpu
#SBATCH --array=0-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers shoppers shoppers shoppers diabetes diabetes diabetes diabetes default default default default)
triggers=(mid mid final mid mid mid final mid mid mid final mid)
labels=(mid25_hard mid40_hard final_hard mid25_broad mid25_hard mid40_hard final_hard mid25_broad mid25_broad mid40_broad final_broad actionable_mid25)
i=$SLURM_ARRAY_TASK_ID; dataset=${datasets[$i]}; trigger=${triggers[$i]}; label=${labels[$i]}
case "$dataset" in shoppers) graph_index=0 ;; default) graph_index=1 ;; diabetes) graph_index=2 ;; esac
sample_dir="$A/outputs/dg_repaint_${dataset}_${label}_${SAMPLE_JOB_ID}_${i}/samples"
method="graph_repaint_${trigger}"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/actionable_union_eval_${dataset}_1060955_${graph_index}/evaluation_graph.json" \
  --sample "tabdiff=$sample_dir/tabdiff.csv" \
  --sample "final_repair=$sample_dir/calibrated_final_repair.csv" \
  --sample "${method}=$sample_dir/${method}.csv" \
  --sample "${method}_plus_repair=$sample_dir/${method}_plus_repair.csv" \
  --seed 20261440 \
  --output "$A/outputs/dg_repaint_eval_${dataset}_${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
