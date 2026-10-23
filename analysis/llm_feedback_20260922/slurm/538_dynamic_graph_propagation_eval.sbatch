#!/usr/bin/env bash
#SBATCH --job-name=llmtd-msg-propeval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SOURCE_JOB_ID=${SOURCE_JOB_ID:?SOURCE_JOB_ID required}
S="$A/outputs/dynamic_messages_propagation_ablation_${SOURCE_JOB_ID}/samples"
CDD="$A/outputs/dynamic_cdd_graph_1069977/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --sample "tabdiff=$S/unconditional.csv" \
  --sample "message_p1=$S/fixed_message_p1.csv" \
  --sample "message_p3=$S/fixed_message_p3.csv" \
  --sample "p1_budget3=$S/fixed_message_p1_budget3.csv" \
  --sample "p3_normalized=$S/fixed_message_p3_normalized.csv" \
  --sample "p3_frozen_routing=$S/fixed_message_p3_frozen_routing.csv" \
  --sample "dynamic_cdd_min_kl=$CDD/dynamic_persistent_calibrated_min_kl.csv" \
  --sample "calibrated_final_repair=$CDD/calibrated_final_repair.csv" \
  --output "$A/outputs/dynamic_messages_propagation_eval_${SLURM_JOB_ID}"
