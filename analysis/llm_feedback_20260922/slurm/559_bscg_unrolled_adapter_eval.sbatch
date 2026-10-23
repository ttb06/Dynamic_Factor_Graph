#!/usr/bin/env bash
#SBATCH --job-name=llmtd-bscg-roll-e
#SBATCH --partition=cpu
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${GRAPH_JOB_ID:?GRAPH_JOB_ID required}"; : "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers shoppers diabetes diabetes); variants=(topology no_topology topology no_topology); graph_indices=(1 1 2 2)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}; variant=${variants[$SLURM_ARRAY_TASK_ID]}; gi=${graph_indices[$SLURM_ARRAY_TASK_ID]}
sample_dir="$A/outputs/bscg_adapter_sample_${dataset}_${variant}_${SAMPLE_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/bootstrap_conditional_graph_${dataset}_${GRAPH_JOB_ID}_${gi}/graph.json" \
  --sample "tabdiff=$sample_dir/tabdiff.csv" --sample "final_repair=$sample_dir/calibrated_final_repair.csv" \
  --sample "graph_conditioned=$sample_dir/graph_conditioned.csv" \
  --sample "graph_conditioned_plus_repair=$sample_dir/graph_conditioned_plus_repair.csv" \
  --seed 20261430 --output "$A/outputs/bscg_adapter_eval_${dataset}_${variant}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
