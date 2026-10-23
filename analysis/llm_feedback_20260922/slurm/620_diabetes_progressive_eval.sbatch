#!/usr/bin/env bash
#SBATCH --job-name=llmtd-diab-prog-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CONTROLLER_JOB_ID:?CONTROLLER_JOB_ID required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
sample_dir="$A/outputs/progressive_graph_controller_diabetes_${CONTROLLER_JOB_ID}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset diabetes \
  --graph "$A/outputs/actionable_union_eval_diabetes_1060955_2/evaluation_graph.json" \
  --sample "tabdiff=$sample_dir/unconditional.csv" \
  --sample "final_repair=$sample_dir/final_reward_calibrated.csv" \
  --sample "proposed=$sample_dir/x0_reward_calibrated_then_final_reward.csv" \
  --seed 20261447 --output "$A/outputs/progressive_graph_controller_eval_diabetes_${SLURM_JOB_ID}"
