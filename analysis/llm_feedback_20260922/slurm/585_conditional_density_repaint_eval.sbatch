#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cond-density-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers shoppers shoppers default default default)
labels=(w025 w05 w1 w025 w05 w1)
eval_indices=(0 0 0 1 1 1)
i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$i]}; label=${labels[$i]}; eval_index=${eval_indices[$i]}
sample_dir="$A/outputs/conditional_density_${dataset}_${label}_${SAMPLE_JOB_ID}_${i}/samples"

cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/actionable_union_eval_${dataset}_1060955_${eval_index}/evaluation_graph.json" \
  --sample "tabdiff=$sample_dir/tabdiff.csv" \
  --sample "final_repair=$sample_dir/calibrated_final_repair.csv" \
  --sample "conditional_density=$sample_dir/conditional_density_repaint.csv" \
  --sample "conditional_density_plus_repair=$sample_dir/conditional_density_repaint_plus_repair.csv" \
  --seed 20261440 \
  --output "$A/outputs/conditional_density_eval_${dataset}_${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
