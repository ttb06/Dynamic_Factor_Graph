#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cond-bootstrap
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers shoppers shoppers default default default)
seeds=(20261444 20261445 20261446 20261444 20261445 20261446)
eval_indices=(0 0 0 1 1 1)
i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$i]}; seed=${seeds[$i]}; eval_index=${eval_indices[$i]}
samples="$A/outputs/conditional_density_confirm_${dataset}_s${seed}_1071209_${i}/samples"

cd "$REPO"
"$PY" "$A/scripts/paired_graph_error_bootstrap.py" \
  --graph "$A/outputs/actionable_union_eval_${dataset}_1060955_${eval_index}/evaluation_graph.json" \
  --heldout "$REPO/synthetic/${dataset}/test.csv" \
  --candidate "$samples/conditional_density_repaint_plus_repair.csv" \
  --baseline "$samples/calibrated_final_repair.csv" \
  --candidate-name conditional_density_plus_repair --baseline-name final_repair \
  --replicates 5000 --seed "$seed" \
  --output "$A/outputs/conditional_density_bootstrap_${dataset}_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
