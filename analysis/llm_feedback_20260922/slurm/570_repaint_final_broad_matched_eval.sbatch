#!/usr/bin/env bash
#SBATCH --job-name=llmtd-repaint-final-be
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers diabetes); eval_indices=(0 2)
i=$SLURM_ARRAY_TASK_ID; dataset=${datasets[$i]}; ei=${eval_indices[$i]}
sample_dir="$A/outputs/dg_repaint_final_broad_${dataset}_${SAMPLE_JOB_ID}_${i}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/actionable_union_eval_${dataset}_1060955_${ei}/evaluation_graph.json" \
  --sample "tabdiff=$sample_dir/tabdiff.csv" --sample "final_repair=$sample_dir/calibrated_final_repair.csv" \
  --sample "graph_repaint_final=$sample_dir/graph_repaint_final.csv" \
  --sample "graph_repaint_final_plus_repair=$sample_dir/graph_repaint_final_plus_repair.csv" \
  --seed 20261440 \
  --output "$A/outputs/dg_repaint_final_broad_eval_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
