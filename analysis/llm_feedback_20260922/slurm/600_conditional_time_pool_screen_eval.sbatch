#!/usr/bin/env bash
#SBATCH --job-name=llmtd-time-pool-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers shoppers default default); policies=(persistent union persistent union); eval_indices=(0 0 1 1)
i=$SLURM_ARRAY_TASK_ID; dataset=${datasets[$i]}; policy=${policies[$i]}; eval_index=${eval_indices[$i]}
sample_dir="$A/outputs/conditional_time_pool_${dataset}_${policy}_${SAMPLE_JOB_ID}_${i}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/actionable_union_eval_${dataset}_1060955_${eval_index}/evaluation_graph.json" \
  --sample "tabdiff=$sample_dir/tabdiff.csv" --sample "final_repair=$sample_dir/calibrated_final_repair.csv" \
  --sample "time_pool_${policy}=$sample_dir/graph_repaint_trajectory.csv" \
  --sample "time_pool_${policy}_plus_repair=$sample_dir/graph_repaint_trajectory_plus_repair.csv" \
  --seed 20261448 --output "$A/outputs/conditional_time_pool_eval_${dataset}_${policy}_${SLURM_ARRAY_JOB_ID}_${i}"
