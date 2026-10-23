#!/usr/bin/env bash
#SBATCH --job-name=llmtd-tabsyn24-e
#SBATCH --partition=prioritized
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${TABSYN_JOB_ID:?TABSYN_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(default shoppers)
graph_indices=(1 0)
i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$i]}; gi=${graph_indices[$i]}; seed=$((20261015 + i))
sample="$A/outputs/tabsyn_official_${dataset}_${TABSYN_JOB_ID}_${i}/tabsyn_4096.csv"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/actionable_union_eval_${dataset}_1060955_${gi}/evaluation_graph.json" \
  --sample "tabsyn_official=$sample" --seed "$seed" \
  --output "$A/outputs/tabsyn_official_eval_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
