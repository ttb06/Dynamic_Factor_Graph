#!/usr/bin/env bash
#SBATCH --job-name=llmtd-msg-certeval
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
S="$A/outputs/dynamic_messages_certainty_gate_${SOURCE_JOB_ID}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --sample "tabdiff=$S/unconditional.csv" \
  --sample "ungated_early=$S/fixed_message_p3.csv" \
  --sample "certainty_tau50=$S/certainty_message_p3_tau50.csv" \
  --sample "certainty_tau75=$S/certainty_message_p3_tau75.csv" \
  --sample "frozen_routing=$S/fixed_message_p3_frozen_routing.csv" \
  --sample "calibrated_final_repair=$S/calibrated_final_repair.csv" \
  --output "$A/outputs/dynamic_messages_certainty_gate_eval_${SLURM_JOB_ID}"
