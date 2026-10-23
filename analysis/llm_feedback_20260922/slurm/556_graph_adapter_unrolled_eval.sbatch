#!/usr/bin/env bash
#SBATCH --job-name=llmtd-gca-roll-e
#SBATCH --partition=cpu
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers shoppers shoppers diabetes diabetes diabetes)
variants=(roll4 roll4 roll4_no_topology roll4 roll4 roll4_no_topology)
seeds=(20261420 20261421 20261420 20261420 20261421 20261420)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}; variant=${variants[$SLURM_ARRAY_TASK_ID]}; seed=${seeds[$SLURM_ARRAY_TASK_ID]}
if [[ "$dataset" == shoppers ]]; then graph=actionable_graph_v2_shoppers_1060952_0/empirical_graph.json
else graph=actionable_graph_v2_diabetes_1060952_2/empirical_graph.json; fi
sample_dir="$A/outputs/graph_adapter_sample_${dataset}_${variant}_s${seed}_${SAMPLE_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/$graph" --sample "tabdiff=$sample_dir/tabdiff.csv" \
  --sample "final_repair=$sample_dir/calibrated_final_repair.csv" \
  --sample "graph_conditioned=$sample_dir/graph_conditioned.csv" \
  --sample "graph_conditioned_plus_repair=$sample_dir/graph_conditioned_plus_repair.csv" \
  --seed "$seed" --output "$A/outputs/graph_adapter_eval_${dataset}_${variant}_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
