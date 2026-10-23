#!/usr/bin/env bash
#SBATCH --job-name=llmtd-repaint-transfer-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
labels=(density1 density2); i=$SLURM_ARRAY_TASK_ID; label=${labels[$i]}
sample_dir="$A/outputs/dg_repaint_transfer_default_${label}_${SAMPLE_JOB_ID}_${i}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" \
  --sample "tabdiff=$sample_dir/tabdiff.csv" --sample "final_repair=$sample_dir/calibrated_final_repair.csv" \
  --sample "graph_repaint_mid=$sample_dir/graph_repaint_mid.csv" \
  --sample "graph_repaint_mid_plus_repair=$sample_dir/graph_repaint_mid_plus_repair.csv" \
  --seed 20261440 \
  --output "$A/outputs/dg_repaint_transfer_eval_default_${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
