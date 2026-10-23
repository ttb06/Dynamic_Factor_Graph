#!/usr/bin/env bash
#SBATCH --job-name=llmtd-nollm-splitev
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SPLIT_JOB_ID=${SPLIT_JOB_ID:?SPLIT_JOB_ID required}
MESSAGE_JOB_ID=${MESSAGE_JOB_ID:?MESSAGE_JOB_ID required}
split_seeds=(20261006 20261007)
split_seed=${split_seeds[$SLURM_ARRAY_TASK_ID]}
GRAPH="$A/outputs/no_llm_graph_split_s${split_seed}_${SPLIT_JOB_ID}_${SLURM_ARRAY_TASK_ID}/graph.json"
S="$A/outputs/no_llm_splitgraph_messages_s${split_seed}_${MESSAGE_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$GRAPH" --sample "tabdiff=$S/unconditional.csv" \
  --sample "data_graph_message=$S/fixed_message_p3.csv" \
  --sample "data_graph_dual_middle=$S/dual_message_p3_middle.csv" \
  --sample "data_graph_final_repair=$S/calibrated_final_repair.csv" \
  --output "$A/outputs/no_llm_splitgraph_eval_s${split_seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
