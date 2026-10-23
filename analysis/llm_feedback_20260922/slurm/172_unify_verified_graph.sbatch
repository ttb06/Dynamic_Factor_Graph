#!/usr/bin/env bash
#SBATCH --job-name=llmtd-unify-graph
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(adult default shoppers); dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
llm_dir=$(find "$A/outputs" -maxdepth 1 -type d -name "nonvacuous_llm_graph_${dataset}_1054225_*" | head -1)
test -n "$llm_dir"
output="$A/outputs/unified_verified_graph_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/unify_verified_specification_graph.py" --repo "$REPO" --dataset "$dataset" \
  --empirical "$A/outputs/cross_dataset_constraint_20260924/all_constraint_candidates.csv" \
  --llm-dir "$llm_dir" --min-support 100 --max-edges 32 --output "$output"
