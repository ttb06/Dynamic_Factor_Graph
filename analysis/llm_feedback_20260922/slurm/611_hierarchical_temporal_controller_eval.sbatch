#!/usr/bin/env bash
#SBATCH --job-name=llmtd-hier-temp-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${HIERARCHICAL_JOB_ID:?HIERARCHICAL_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(default shoppers); eval_indices=(1 0)
i=$SLURM_ARRAY_TASK_ID; dataset=${datasets[$i]}; eval_index=${eval_indices[$i]}
sample_dir="$A/outputs/hierarchical_temporal_${dataset}_${HIERARCHICAL_JOB_ID}_${i}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/actionable_union_eval_${dataset}_1060955_${eval_index}/evaluation_graph.json" \
  --sample "tabdiff=$sample_dir/unconditional.csv" \
  --sample "final_repair=$sample_dir/final_reward_calibrated.csv" \
  --sample "hierarchical_online=$sample_dir/x0_reward_calibrated.csv" \
  --sample "hierarchical_plus_repair=$sample_dir/x0_reward_calibrated_then_final_reward.csv" \
  --seed 20261447 --output "$A/outputs/hierarchical_temporal_eval_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
