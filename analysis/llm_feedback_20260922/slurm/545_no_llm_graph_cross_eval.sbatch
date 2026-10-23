#!/usr/bin/env bash
#SBATCH --job-name=llmtd-nollm-xeval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
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
NO_LLM_JOB_ID=${NO_LLM_JOB_ID:?NO_LLM_JOB_ID required}
HEADLINE_JOB_ID=${HEADLINE_JOB_ID:?HEADLINE_JOB_ID required}
seeds=(20261376 20261377 20261378)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}
D="$A/outputs/no_llm_messages_s${seed}_${NO_LLM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples"
H="$A/outputs/headline8_messages_s${seed}_${HEADLINE_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples"
cd "$REPO"
common=(
  --repo "$REPO" --dataset default
  --sample "tabdiff=$D/unconditional.csv"
  --sample "data_graph_message=$D/fixed_message_p3.csv"
  --sample "headline_graph_message=$H/fixed_message_p3.csv"
  --sample "data_graph_dual_middle=$D/dual_message_p3_middle.csv"
  --sample "headline_graph_dual_middle=$H/dual_message_p3_middle.csv"
  --sample "data_graph_final_repair=$D/calibrated_final_repair.csv"
  --sample "headline_graph_final_repair=$H/calibrated_final_repair.csv"
)
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" "${common[@]}" \
  --graph "$A/outputs/no_llm_empirical_graph_20261012/graph.json" \
  --output "$A/outputs/no_llm_cross_eval_data_graph_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" "${common[@]}" \
  --graph "$A/outputs/natural_dag_multi_entry_v3_20261010/graph.json" \
  --output "$A/outputs/no_llm_cross_eval_headline_graph_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
