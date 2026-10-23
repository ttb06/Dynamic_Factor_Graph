#!/usr/bin/env bash
#SBATCH --job-name=llmtd-msg-eval
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
SOURCE_JOB_ID=${SOURCE_JOB_ID:?SOURCE_JOB_ID required}
labels=(weak medium strong)
label=${labels[$SLURM_ARRAY_TASK_ID]}
S="$A/outputs/dynamic_messages_${label}_${SOURCE_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples"
CDD="$A/outputs/dynamic_cdd_graph_1069977/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --sample "tabdiff=$S/unconditional.csv" \
  --sample "fixed_message_p1=$S/fixed_message_p1.csv" \
  --sample "fixed_message_p3=$S/fixed_message_p3.csv" \
  --sample "dual_message_p1=$S/dual_message_p1.csv" \
  --sample "dual_message_p3=$S/dual_message_p3.csv" \
  --sample "dual_message_p3_middle=$S/dual_message_p3_middle.csv" \
  --sample "dual_message_p3_remask=$S/dual_message_p3_remask.csv" \
  --sample "dynamic_cdd_min_kl=$CDD/dynamic_persistent_calibrated_min_kl.csv" \
  --sample "calibrated_final_repair=$CDD/calibrated_final_repair.csv" \
  --output "$A/outputs/dynamic_messages_eval_${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
