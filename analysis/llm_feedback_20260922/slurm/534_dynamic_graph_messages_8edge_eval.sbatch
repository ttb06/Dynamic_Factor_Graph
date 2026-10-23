#!/usr/bin/env bash
#SBATCH --job-name=llmtd-msg-8edgeev
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
S="$A/outputs/dynamic_messages_8edge_${SOURCE_JOB_ID}/samples"
OLD="$A/outputs/default_8edge_frozen_s20261376_1067698_0"
GRAPH="$A/outputs/natural_dag_multi_entry_v3_20261010/graph.json"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$GRAPH" \
  --sample "tabdiff=$S/unconditional.csv" \
  --sample "fixed_message_p1=$S/fixed_message_p1.csv" \
  --sample "fixed_message_p3=$S/fixed_message_p3.csv" \
  --sample "dual_message_p3_middle=$S/dual_message_p3_middle.csv" \
  --sample "sumproduct_message_p1=$S/sumproduct_message_p1.csv" \
  --sample "sumproduct_message_p3=$S/sumproduct_message_p3.csv" \
  --sample "previous_xstate_reward=$OLD/samples/xstate_reward_calibrated.csv" \
  --sample "calibrated_final_repair=$OLD/samples/final_reward_calibrated.csv" \
  --output "$A/outputs/dynamic_messages_8edge_eval_${SLURM_JOB_ID}"
